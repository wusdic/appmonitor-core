"""Read projections shared by the legacy routes (routes.py) and API v2
(routes_v2.py), docs/lib3/api_ui.md.

Everything here only READS the store, through the public MetricStore API and
the lib/m_*.py accessors, never through engine internals. Every projection is
defensive: a model or series that does not exist (yet) yields None / [] / {},
never an exception, because engines land and refit on their own cadence (and
B29 explain may not be registered at all).
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import math
import time
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..engines.behavior.lib import m_class, m_feedback, m_governor, m_portrait, m_rhythm
from ..engines.behavior.lib.classkeys import class_id, class_kind, is_class, is_pseudo
from ..engines.behavior.lib.detectors import DETECTORS, FAMILIES, family_of
from ..engines.behavior.lib.features import (FEATURE_GROUP, FEATURE_KIND, FEATURE_NAMES_V2)
from .serialize import to_jsonable

DAY = 86400.0
DRIFT_DETECTORS = ("cusum", "mcusum", "bocpd", "creep")
LIVE_STATUSES = ("open", "acked")
TIERS = ("low", "medium", "high", "critical")
SURPRISE_CAP = 40.0            # log10(1/e_day) shown in bars (e_day 1e-40)


# =================================================================== runtime
def runtime_config(r: Any) -> Dict[str, Any]:
    cfg = getattr(r, "config", None)
    if not isinstance(cfg, Mapping):
        cfg = getattr(getattr(r, "pipeline", None), "config", None)
    return dict(cfg) if isinstance(cfg, Mapping) else {}


def runtime_tz(r: Any) -> str:
    return str(runtime_config(r).get("tz") or "Asia/Shanghai")


def store_now(r: Any) -> float:
    """The pipeline's clock: the virtual time the last tick was stamped with
    (contract I: every tick runs at now = gen.vt). Falls back to the newest
    engine health record, then wall clock."""
    vt = getattr(getattr(r, "gen", None), "vt", None)
    if isinstance(vt, (int, float)) and math.isfinite(vt):
        return float(vt)
    try:
        ts = [float(h.get("ts")) for h in r.store.health().values() if h.get("ts") is not None]
        if ts:
            return max(ts)
    except Exception:
        pass
    return time.time()


def window_s(r: Any) -> float:
    try:
        return float(getattr(r, "window_s", None) or getattr(r.pipeline, "window_s", 60))
    except Exception:
        return 60.0


@contextlib.contextmanager
def read_lock(r: Any, timeout: float = 5.0) -> Iterator[None]:
    """Hold the runtime's tick lock while projecting, so a live tick cannot
    mutate a model dict mid-read. Never blocks a request for long: after
    `timeout` seconds the read proceeds unlocked (the store itself is
    thread-safe; only engine-owned dicts could be mid-update)."""
    lock = getattr(r, "_lock", None)
    got = False
    if lock is not None:
        try:
            got = lock.acquire(timeout=timeout)
        except Exception:
            got = False
    try:
        yield
    finally:
        if got:
            lock.release()


def iso(ts: Any, tz: str) -> Optional[str]:
    """ISO-8601 local time in the configured tz (None for missing ts)."""
    x = fnum(ts)
    if x is None:
        return None
    try:
        from zoneinfo import ZoneInfo
        return _dt.datetime.fromtimestamp(x, ZoneInfo(tz)).isoformat(timespec="seconds")
    except Exception:
        return _dt.datetime.fromtimestamp(x, _dt.timezone.utc).isoformat(timespec="seconds")


def fnum(x: Any) -> Optional[float]:
    """float(x) when finite, else None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def rnd(x: Any, nd: int = 4) -> Optional[float]:
    v = fnum(x)
    return round(v, nd) if v is not None else None


def sev_name(x: Any) -> str:
    v = getattr(x, "value", x)
    return str(v).lower() if v is not None else ""


# ================================================================ series reads
def latest_vec(store: Any, s: str, e: str, name: str) -> Tuple[Optional[float], Optional[np.ndarray]]:
    ts, M = store.vec_tail(s, e, name, 1)
    if not len(ts):
        return None, None
    return float(ts[-1]), np.asarray(M[-1], dtype=np.float64)


def latest_scalar(store: Any, s: str, e: str, name: str) -> Optional[float]:
    """Newest value of a 1-element vec ring (or a scalar derived series)."""
    _, row = latest_vec(store, s, e, name)
    if row is not None and row.size:
        return fnum(row[0])
    d = store.latest_derived(s, e, name)
    return fnum(d.value) if d is not None and not isinstance(d.value, dict) else None


def latest_dict(store: Any, s: str, e: str, name: str) -> Tuple[Optional[float], Dict[str, Any]]:
    d = store.latest_derived(s, e, name)
    if d is not None and isinstance(d.value, dict):
        return float(d.ts), dict(d.value)
    return None, {}


def _thin(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def scalar_series(store: Any, s: str, e: str, name: str, since: float,
                  max_points: int = 1500) -> Dict[str, List]:
    """{ts, values} of a 1-element vec ring since `since` (thinned evenly)."""
    ts, M = store.vec_since(s, e, name, since)
    if not len(ts):
        return {"ts": [], "values": []}
    idx = _thin(len(ts), max_points)
    return {"ts": [float(t) for t in ts[idx]],
            "values": [fnum(v) for v in np.asarray(M[idx, 0], dtype=np.float64)]}


def dict_series(store: Any, s: str, e: str, name: str, since: float,
                max_points: int = 1500) -> List[Tuple[float, Dict[str, Any]]]:
    rows = [m for m in store.derived_series(s, e, name)
            if m.ts >= since and isinstance(m.value, dict)]
    idx = _thin(len(rows), max_points)
    return [(float(rows[i].ts), dict(rows[i].value)) for i in idx]


# ===================================================================== risk
def profile_extra(store: Any, s: str, e: str) -> Dict[str, Any]:
    p = store.profile(s, e)
    return dict(p.extra or {}) if p is not None else {}


def risk_info(store: Any, s: str, e: str, extra: Optional[Mapping] = None) -> Dict[str, Any]:
    """Current risk: the latest behavior.risk row (B26) plus tier / trend /
    reasons from profile.extra.risk. Tier names are lowercase (contract B)."""
    extra = profile_extra(store, s, e) if extra is None else extra
    ex = dict(extra.get("risk") or {})
    score = latest_scalar(store, s, e, "behavior.risk")
    if score is None:
        score = fnum(ex.get("score"))
    tier = str(ex.get("tier") or "").lower() or (tier_of(score) if score is not None else None)
    return {"score": rnd(score, 2), "tier": tier, "trend": rnd(ex.get("trend"), 2),
            "top_reasons": list(ex.get("top_reasons") or []),
            "stages": list(ex.get("stages") or []), "degraded": ex.get("degraded")}


def tier_of(score: Optional[float]) -> Optional[str]:
    """B26 thresholds (without its hysteresis) for keys that have no profile."""
    if score is None:
        return None
    return "critical" if score >= 85 else "high" if score >= 60 else \
        "medium" if score >= 30 else "low"


def e_day_now(store: Any, s: str, e: str, dt_s: float) -> Optional[float]:
    """behavior.e_day of the newest fusion row, else q_all * 86400 / dt."""
    v = latest_scalar(store, s, e, "behavior.e_day")
    if v is not None:
        return v
    q = latest_scalar(store, s, e, "behavior.q_all")
    return q * DAY / dt_s if q is not None else None


def anomaly_score(store: Any, s: str, e: str, dt_s: float) -> float:
    """Legacy field (api_ui.md): min(1, log10(1 / max(e_day, 1e-12)) / 8) of
    the latest q_all's e_day; 0 when unscored or e_day >= 1."""
    ed = e_day_now(store, s, e, dt_s)
    if ed is None:
        return 0.0
    return float(min(1.0, max(0.0, math.log10(1.0 / max(ed, 1e-12)) / 8.0)))


def drift_score(store: Any, s: str, e: str) -> float:
    """Legacy field: max over {cusum, mcusum, bocpd, creep} of min(1, score/10)
    on the newest behavior.score row."""
    _, row = latest_vec(store, s, e, "behavior.score")
    if row is None or row.size != len(DETECTORS):
        return 0.0
    best = 0.0
    for d in DRIFT_DETECTORS:
        v = fnum(row[DETECTORS.index(d)])
        if v is not None:
            best = max(best, min(1.0, max(0.0, v / 10.0)))
    return float(best)


# ============================================================ classes / roles
def peer_group(store: Any, s: str, e: str, extra: Optional[Mapping] = None) -> Dict[str, Any]:
    """profile.extra.peer_group (B02, contract G), else model.class's assignment."""
    extra = profile_extra(store, s, e) if extra is None else extra
    pg = dict(extra.get("peer_group") or {})
    if not pg:
        try:
            a = m_class.assignment(store, s, e) or {}
        except Exception:
            a = {}
        if a:
            pg = {"role": a.get("role"), "sub": a.get("sub"), "prob": a.get("prob"),
                  "static_classes": list(a.get("static") or []), "pool": a.get("pool"),
                  "class_path": a.get("class_path"), "super": a.get("super"),
                  "provisional": a.get("provisional")}
            try:
                pg["role_name"] = m_class.role_name(store, a.get("role")) if a.get("role") else None
            except Exception:
                pg["role_name"] = None
    return pg


def regime_info(store: Any, s: str, e: str) -> Dict[str, Any]:
    try:
        d = m_governor.descriptor(store, s, e)
    except Exception:
        d = {}
    _, rg = latest_dict(store, s, e, "behavior.regime")
    out = dict(d or {})
    if rg:
        out.setdefault("state", rg.get("state"))
        out["live"] = rg
    return out


def open_incident_count(store: Any, s: str, e: str) -> int:
    try:
        return len(store.incidents(system=s, entity=e, status=LIVE_STATUSES))
    except Exception:
        return 0


# ================================================================ incidents
def rare_once_in_days(e_day: Any) -> Optional[float]:
    """e_day = expected equally extreme null ticks per entity-day, so the
    finding is 'as rare as once in 1/e_day days for this entity'."""
    v = fnum(e_day)
    if v is None or v <= 0:
        return None
    return 1.0 / v


def surprise(e_day: Any) -> Optional[float]:
    v = fnum(e_day)
    if v is None:
        return None
    return float(min(SURPRISE_CAP, max(0.0, -math.log10(max(v, 1e-300)))))


def incident_summary(inc: Any, tz: str) -> Dict[str, Any]:
    ent = inc.entity or ""
    new = _incident_new_tokens(inc)
    return {
        "id": inc.id, "system": inc.system, "entity": ent,
        "kind": "class" if is_class(ent) else "entity",
        "entities": list(inc.entities or []), "kinds": list(inc.kinds or []),
        "axes": list(inc.axes or []), "status": inc.status,
        "severity": sev_name(inc.severity), "opened": inc.opened, "last_seen": inc.last_seen,
        "opened_local": iso(inc.opened, tz), "last_seen_local": iso(inc.last_seen, tz),
        "e_day_min": fnum(inc.e_day_min), "rare_once_in_days": rare_once_in_days(inc.e_day_min),
        "risk": rnd(inc.risk, 2), "campaign_id": inc.campaign_id or None,
        "parent_id": inc.parent_id or None, "close_reason": inc.close_reason,
        "narrative": narrative(inc), "new_tokens": new[:12],
        "n_evidence": len(inc.evidence or []),
    }


def narrative(inc: Any) -> Dict[str, Optional[str]]:
    """{zh, en, headline_zh, headline_en}: B29 explain writes narrative_zh /
    narrative_en / headline_zh / headline_en into incident.explanation (an
    older writer may nest {'zh', 'en'} under 'narrative' or set attributes).
    Incident.narrative (B29 copies the zh text there) is the fallback only
    when the explanation carries neither language, so an English reader is
    never shown the Chinese text labelled as English."""
    ex = inc.explanation if isinstance(getattr(inc, "explanation", None), Mapping) else {}
    base = getattr(inc, "narrative", "") or ""
    zh = getattr(inc, "narrative_zh", None) or ex.get("narrative_zh")
    en = getattr(inc, "narrative_en", None) or ex.get("narrative_en")
    nar = ex.get("narrative")
    if isinstance(nar, Mapping):
        zh = zh or nar.get("zh")
        en = en or nar.get("en")
    if not zh and not en:
        zh = en = base or None
    return {"zh": zh or None, "en": en or None,
            "headline_zh": ex.get("headline_zh") or None,
            "headline_en": ex.get("headline_en") or None}


def mask_template(tok: Any) -> Any:
    """A template / HTTP token as B30 publishes it: every path or query VALUE
    re-masked (lib/m_portrait.safe_token); None when there is nothing to show."""
    return None if tok is None else m_portrait.safe_token(tok)


def portrait_diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Material changes between two kept portrait versions with B30's own
    semantics (lib/m_portrait.signature / diff_signatures); a version without
    a stored 'sig' is signed from its json."""
    sa = a.get("sig") if isinstance(a.get("sig"), Mapping) else \
        m_portrait.signature(a.get("json") or {})
    sb = b.get("sig") if isinstance(b.get("sig"), Mapping) else \
        m_portrait.signature(b.get("json") or {})
    return m_portrait.diff_signatures(sa, sb)


def _incident_new_tokens(inc: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for ev in inc.evidence or []:
        if not isinstance(ev, Mapping):
            continue
        for t in ev.get("new_tokens") or []:
            if t not in seen:
                seen.add(t)
                out.append(str(t))
    ex = inc.explanation if isinstance(inc.explanation, Mapping) else {}
    for t in ex.get("new_tokens") or []:
        tok = t.get("token") if isinstance(t, Mapping) else t
        if tok is not None and str(tok) not in seen:
            seen.add(str(tok))
            out.append(str(tok))
    return out


def evidence_bars(inc: Any) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Per-family and per-axis bars: the strongest surprise log10(1/e_day)
    seen in the incident's evidence (alarms by their families, discrete
    findings by their contract family, detector p-values by detector family)."""
    fam: Dict[str, Dict[str, Any]] = {}
    axis: Dict[str, Dict[str, Any]] = {}

    def bump(d: Dict[str, Dict[str, Any]], key: str, e_day: Any, src: str) -> None:
        s_ = surprise(e_day)
        if not key or s_ is None:
            return
        cur = d.setdefault(key, {"surprise": 0.0, "e_day": None, "n": 0, "sources": []})
        cur["n"] += 1
        if src not in cur["sources"]:
            cur["sources"].append(src)
        if s_ >= cur["surprise"]:
            cur["surprise"] = s_
            cur["e_day"] = fnum(e_day)

    for ev in inc.evidence or []:
        if not isinstance(ev, Mapping):
            continue
        src = str(ev.get("source") or "")
        ed = ev.get("e_day")
        if ed is None and ev.get("severity"):
            ed = m_feedback.SEVERITY_E_DAY.get(sev_name(ev.get("severity")))
        fams = list(ev.get("families") or [])
        if src == "event" and ev.get("kind") in m_feedback.EVENT_FAMILY:
            fams.append(m_feedback.EVENT_FAMILY[ev["kind"]])
        for f in fams:
            bump(fam, str(f), ed, src)
        for a in ev.get("axes") or []:
            bump(axis, str(a), ed, src)
    for d in (fam, axis):
        for v in d.values():
            v["surprise"] = round(v["surprise"], 3)
    return fam, axis


def counterfactual(inc: Any) -> Dict[str, Any]:
    ex = inc.explanation if isinstance(inc.explanation, Mapping) else {}
    cf = ex.get("counterfactual")
    cf = dict(cf) if isinstance(cf, Mapping) else {}
    return {"set": cf.get("set", ex.get("counterfactual_set")),
            "valid": cf.get("valid", ex.get("counterfactual_valid")),
            "scope": cf.get("scope", ex.get("counterfactual_scope")),
            "available": bool(cf or "counterfactual_set" in ex)}


def label_view(lb: Any, tz: str) -> Dict[str, Any]:
    d = to_jsonable(lb)
    d["ts_local"] = iso(lb.ts, tz)
    return d


def event_view(e: Any, tz: str) -> Dict[str, Any]:
    d = to_jsonable(e)
    d["ts_local"] = iso(e.ts, tz)
    return d


# ================================================================== rhythm
DOW_EPOCH = 3                    # 1970-01-01 (local epoch day 0) was a Thursday


def rhythm_grid(store: Any, s: str, e: str, now: float, tz: str,
                calendar: Any = None) -> Optional[Dict[str, Any]]:
    """7x24 heatmap of P(active) per hour (mean of the four 15-min slot
    cells of model.rhythm's 168 profile, m_rhythm.profile168), the observed
    activity of the last 7 days from the rhythm ledger, and where 'now' is."""
    model = store.get_model(s, e, "model.rhythm")
    if not isinstance(model, Mapping) or not isinstance(model.get("state"), Mapping):
        return None
    try:
        p168 = np.asarray(m_rhythm.profile168(model), dtype=np.float64)
        grid = p168.reshape(7, 24, 4).mean(axis=2)
    except Exception:
        return None
    obs_sum = np.zeros((7, 24))
    obs_n = np.zeros((7, 24))
    try:
        for slot, a, _vol, _c48, _c168 in m_rhythm.slot_history(model, since=now - 7 * DAY):
            if a != a:
                continue
            loc = int(slot) * 900
            day, sec = divmod(loc, 86400)
            dow = (day + DOW_EPOCH) % 7
            obs_sum[dow, sec // 3600] += float(a)
            obs_n[dow, sec // 3600] += 1.0
    except Exception:
        pass
    observed = np.where(obs_n > 0, obs_sum / np.maximum(obs_n, 1.0), np.nan)
    cur: Dict[str, Any] = {}
    try:
        from ..engines.behavior.lib import timebins
        t = timebins.tctx(now, tz, timebins.parse_calendar(calendar) if calendar else None)
        cur = {"dow": int(t["dow"]), "hour": int(math.floor(float(t["hour_local"]))),
               "day_type": t.get("day_type")}
    except Exception:
        pass
    cur["active"] = latest_scalar(store, s, e, "feature.active")
    try:
        mature = bool(m_rhythm.mature168(model))
    except Exception:
        mature = None
    return {"p_active": to_jsonable(np.round(grid, 4)), "observed_7d": to_jsonable(np.round(observed, 3)),
            "now": cur, "mature168": mature, "dow_labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "source": "m_rhythm.profile168 (hourly mean of 15-min cells)"}


# ================================================================ features
_GRAIN_UNIT = {"h": "per 60 min", "q": "per 15 min"}
_GRAIN_Z = {"h": ("behavior.z", "behavior.zr"), "q": ("behavior.z.q", "behavior.zr.q")}


def feature_rows(store: Any, s: str, e: str, extra: Mapping[str, Any],
                 prof: Any, dt_s: float) -> List[Dict[str, Any]]:
    """Per-feature rows: the current value and the predictive p5/p50/p95 in
    natural units (profile.extra.model_state, B04), plus the engine's own z
    (behavior.z, current anchor) and zr (reference anchor). Count and bytes
    features are shown at the model_state exposure (15 min) so current and
    band are comparable; `current_tick` is the raw per-tick value.

    spec v2.1 (cadence.md §9.5): with grain model_state each row gains
    grains: {h: {current, p5, p50, p95, z, zr}, q: {..., provisional}}, where
    `current` is the live rolling grain value (feature.live.<g>, exact for
    distinct counts; v2 scaled the tick value by 900 / Δt, which is wrong for
    sub-additive features). The legacy fields are filled from Q when Q is
    observable, else from H, and `unit` names the grain."""
    ms = extra.get("model_state") if isinstance(extra.get("model_state"), Mapping) else {}
    feats = ms.get("features") if isinstance(ms.get("features"), Mapping) else {}
    ms_dt = fnum(ms.get("dt_s")) or 900.0
    names = list(getattr(prof, "feature_names", None) or FEATURE_NAMES_V2)
    _, nat = latest_vec(store, s, e, "feature.nat")
    gms = ms.get("grains") if isinstance(ms.get("grains"), Mapping) else {}
    live: Dict[str, Any] = {}
    gz: Dict[str, Tuple[Any, Any]] = {}
    for g in ("h", "q"):
        if not isinstance(gms.get(g), Mapping):
            continue
        tl, lv = latest_vec(store, s, e, f"feature.live.{g}")
        live[g] = lv
        gz[g] = (latest_vec(store, s, e, _GRAIN_Z[g][0])[1],
                 latest_vec(store, s, e, _GRAIN_Z[g][1])[1])
    # the legacy fields: Q when Q is observable (a live Q row), else H
    leg = "q" if live.get("q") is not None else ("h" if "h" in live else None)
    if leg is not None:
        feats = gms[leg].get("features") if isinstance(gms[leg].get("features"), Mapping) else {}
        ms_dt = fnum(gms[leg].get("dt_s")) or ms_dt
        z, zr = gz[leg]
    else:
        _, z = latest_vec(store, s, e, "behavior.z")
        _, zr = latest_vec(store, s, e, "behavior.zr")
    rows = []
    for i, name in enumerate(names):
        kind = FEATURE_KIND.get(name, "")
        cur_tick = fnum(nat[i]) if nat is not None and i < nat.size else None
        if leg is not None:
            lv = live.get(leg)
            cur = fnum(lv[i]) if lv is not None and i < lv.size else None
        else:
            scale = ms_dt / dt_s if kind in ("count", "bytes") and dt_s > 0 else 1.0
            cur = cur_tick * scale if cur_tick is not None else None
        q = feats.get(name)
        p5 = p50 = p95 = None
        if isinstance(q, (list, tuple)) and len(q) >= 3:
            p5, p50, p95 = fnum(q[0]), fnum(q[1]), fnum(q[2])
        zi = fnum(z[i]) if z is not None and i < z.size else None
        zri = fnum(zr[i]) if zr is not None and i < zr.size else None
        spread = (p95 - p5) / 3.29 if p5 is not None and p95 is not None else None
        unit = None
        if kind in ("count", "bytes"):
            unit = _GRAIN_UNIT.get(leg, "per 15 min") if leg else "per 15 min"
        row = {
            "name": name, "group": FEATURE_GROUP.get(name), "kind": kind,
            "unit": unit,
            "current": rnd(cur, 4), "current_tick": rnd(cur_tick, 4),
            "p5": rnd(p5, 4), "p50": rnd(p50, 4), "p95": rnd(p95, 4),
            "z": rnd(zi, 3), "zr": rnd(zri, 3),
            # legacy (routes.py v1) field names, same semantics
            "baseline": rnd(p50, 4), "spread": rnd(spread, 4), "stable": rnd(p50, 4),
        }
        if live:
            grains: Dict[str, Any] = {}
            for g, lv in live.items():
                blk = gms[g]
                gf = blk.get("features") if isinstance(blk.get("features"), Mapping) else {}
                gq = gf.get(name)
                b5 = b50 = b95 = None
                if isinstance(gq, (list, tuple)) and len(gq) >= 3:
                    b5, b50, b95 = fnum(gq[0]), fnum(gq[1]), fnum(gq[2])
                zg, zrg = gz[g]
                gr = {"current": rnd(fnum(lv[i]) if lv is not None and i < lv.size else None, 4),
                      "p5": rnd(b5, 4), "p50": rnd(b50, 4), "p95": rnd(b95, 4),
                      "z": rnd(fnum(zg[i]) if zg is not None and i < zg.size else None, 3),
                      "zr": rnd(fnum(zrg[i]) if zrg is not None and i < zrg.size else None, 3),
                      "unit": _GRAIN_UNIT[g] if kind in ("count", "bytes") else None}
                if g == "q":
                    gr["provisional"] = bool(blk.get("provisional", False))
                grains[g] = gr
            row["grains"] = grains
        rows.append(row)
    return rows


# ================================================================= classes
def class_catalog(store: Any, s: str) -> List[Dict[str, Any]]:
    """Every class of a system: role / static / pool pseudo-entities (plus
    role classes model.class knows with >= 1 member here) and the role
    sub-classes (subs have no pseudo-entity; they are listed under their role)."""
    try:
        m = m_class.get(store) or {}
    except Exception:
        m = {}
    keys = set(k for k in store.pseudo_entities(s) if is_class(k))
    try:
        keys.update(m_class.all_class_keys(store, s, min_members=1))
    except Exception:
        pass
    assign = m.get("assign") or {}
    roles = m.get("roles") or {}
    out = []
    for key in sorted(keys):
        kind = class_kind(key) or "role"
        cid = class_id(key)
        try:
            mem = m_class.class_members(store, s, key)
        except Exception:
            mem = []
        name = cid
        if kind == "role":
            name = (roles.get(cid) or {}).get("name") or cid
        out.append({"key": key, "kind": kind, "id": cid, "name": name, "parent": None,
                    "super": (roles.get(cid) or {}).get("super") if kind == "role" else None,
                    "members": mem, "n_members": len(mem)})
    subs = m.get("subs") or {}
    for sid, sub in sorted(subs.items()):
        mem = sorted(x.partition("|")[2] for x in (sub.get("members") or [])
                     if x.partition("|")[0] == s)
        if not mem:
            continue
        rid = sub.get("role")
        out.append({"key": f"sub:{sid}", "kind": "sub", "id": sid, "name": sub.get("name") or sid,
                    "parent": f"class:{rid}" if rid else None,
                    "super": (roles.get(rid) or {}).get("super"),
                    "members": mem, "n_members": len(mem)})
    # members' membership probability
    for c in out:
        c["member_probs"] = {ip: fnum((assign.get(f"{s}|{ip}") or {}).get("prob"))
                             for ip in c["members"]}
    return out


def class_tree(classes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """super -> role -> sub hierarchy for the class page."""
    roles = [c for c in classes if c["kind"] == "role"]
    subs = [c for c in classes if c["kind"] == "sub"]
    supers: Dict[str, List[Dict[str, Any]]] = {}
    for r in roles:
        node = {"key": r["key"], "name": r["name"], "n_members": r["n_members"],
                "subs": [{"key": sc["key"], "name": sc["name"], "n_members": sc["n_members"],
                          "members": sc["members"]}
                         for sc in subs if sc["parent"] == r["key"]]}
        supers.setdefault(r.get("super") or "unknown", []).append(node)
    tree = [{"super": k, "roles": v} for k, v in sorted(supers.items())]
    others = [{"key": c["key"], "kind": c["kind"], "name": c["name"], "n_members": c["n_members"]}
              for c in classes if c["kind"] in ("static", "pool")]
    if others:
        tree.append({"super": "static/pool", "roles": others})
    return tree


def resolve_class_key(cid: str) -> str:
    """Accept 'class:r2', 'r2', 'static:<n>', 'pool:<cidr>' or 'sub:<sid>'."""
    if cid.startswith("class:") or cid.startswith("sub:"):
        return cid
    return "class:" + cid


# ========================================================= detectors health
SERIES_PREFIXES = ("behavior.", "feature.", "derived.", "ops.", "l2.", "l3.", "l4.", "http.",
                   "dns.", "tls.", "probe.", "act.", "client.")


def produced_series(produces: Iterable[str]) -> List[str]:
    """The concrete series names an engine declares (skipping models,
    profile fields, events, checkpoints and wildcard templates)."""
    out = []
    for p in produces or []:
        p = str(p)
        if any(ch in p for ch in "*<{:") or not p.startswith(SERIES_PREFIXES):
            continue
        out.append(p)
    return out


def last_write_any(store: Any, name: str, keys: Sequence[Tuple[str, str]]) -> Optional[float]:
    best = None
    for s, e in keys:
        t = store.last_write_ts(s, e, name)
        if t is not None and (best is None or t > best):
            best = t
    return best


def degraded_counts(store: Any, s: str) -> Dict[str, Dict[str, Any]]:
    """{detector: {'n', 'degraded', 'causes': {kind: count}}} over the keys of
    system s (entities and class keys) at each key's last behavior.score
    tick: n = keys whose detector ran there (a finite score, or an entry in
    behavior.degraded at that tick), degraded = those with an entry; the
    cause kind is the part before ':' (lib/emit.CAUSES)."""
    out: Dict[str, Dict[str, Any]] = {}
    keys = list(store.entities(s)) + [k for k in store.pseudo_entities(s) if is_class(k)]
    for e in keys:
        ts, row = latest_vec(store, s, e, "behavior.score")
        if ts is None or row is None:
            continue
        t_d, dg = latest_dict(store, s, e, "behavior.degraded")
        dg = dg if (dg and t_d == ts) else {}
        vals = np.asarray(row, dtype=np.float64).reshape(-1)
        for i, d in enumerate(DETECTORS):
            ran = (i < vals.size and math.isfinite(float(vals[i]))) or d in dg
            if not ran:
                continue
            c = out.setdefault(d, {"n": 0, "degraded": 0, "causes": {}})
            c["n"] += 1
            if d in dg:
                c["degraded"] += 1
                kind = str(dg[d]).split(":", 1)[0]
                c["causes"][kind] = c["causes"].get(kind, 0) + 1
    return out


def all_keys(store: Any) -> List[Tuple[str, str]]:
    keys = []
    for s in store.systems():
        for e in store.entities(s, include_pseudo=True):
            keys.append((s, e))
    return keys


def detector_family_map() -> Dict[str, str]:
    return {d: family_of(d) for d in DETECTORS}
