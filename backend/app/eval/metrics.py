"""Evaluation metrics and the 16 acceptance gates (docs/lib3/eval.md).

Two stages, so that the expensive part parallelises:

  score_run(run)        one RunResult -> a JSON-able dict of per-run
                        measurements (per-scenario outcomes, FAR counts,
                        calibration statistics, ...). Runs inside the worker
                        process, next to the data it needs.
  compute_gates(scores) the list of per-run dicts (5 seeds x 5 packs) -> the
                        16 gates, each {value, target, pass, details}, with
                        medians over seeds and bootstrap 95% CIs.

Ground truth comes only from run.truth (gen.truth); the pipeline never saw
it. Everything the system *claims* is read from what it published:
incidents (severity history recorded by the runner, axes, kinds, evidence,
explanation), discrete events, profile.extra, behavior.* series and the
model version history. Scoring rules from eval.md "LABELS":

  * a detector or axis counts only through the incident's axes / detectors
    (p_by_detector in evidence, explanation or linked events) or through
    discrete events on the scenario entity;
  * the scenario entity is expanded with its continuity aliases, its actor
    chain and, for class scenarios, its class key;
  * control entities are those with no scenario in the pack; their exposure
    excludes [t_start - 1 h, t_end + 24 h] of any scenario whose class (or,
    for system-wide changes, system) includes them.

`pass` is True / False, or None when the gate (or a sub-check) cannot be
computed from the data at hand (e.g. an engine that has not been written yet
publishes nothing). A gate passes only if every computable sub-check passes.

Range-based precision / recall / F1 follow Tatbul et al., "Precision and
Recall for Time Series" (NeurIPS 2018), in continuous time with an existence
weight alpha, a positional bias (flat / front / back / middle) and the
reciprocal cardinality factor; point-adjusted F1 is reported for reference.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from ..engines.behavior.lib.detectors import (BUDGET_PATHS, DETECTOR_INFO, DETECTORS,
                                                  FAMILIES, FAMILY_MEMBERS)
except Exception:  # pragma: no cover - the lib is part of every checkout
    BUDGET_PATHS, DETECTOR_INFO, DETECTORS, FAMILIES, FAMILY_MEMBERS = {}, {}, [], [], {}

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
SEVERITIES = ("info", "low", "medium", "high", "critical")
SEV_RANK = {s: i for i, s in enumerate(SEVERITIES)}

LOUD = frozenset({"T1", "T6", "T6b", "T9", "T11", "T12"})
SUBTLE = frozenset({"T2", "T3", "T4", "T4b", "T5", "T7", "T8", "T9b", "T10", "T13", "T14",
                    "T15", "T16", "T17", "T18", "T19", "T21"})
P2_SCENARIOS = frozenset({"T20"})
CLASS_SCENARIOS = frozenset({"T21", "L1", "L2", "L3", "L10"})
ACTOR_SCENARIOS = frozenset({"T19"})
CLASS_LEGIT = frozenset({"L1", "L2", "L3", "L10"})
PACK_E_REPLAYS = ("T1", "T5", "T12")

HUMAN_ARCHETYPES = frozenset({"interactive", "search", "hr", "human", "nat", "explorer"})
DEFAULT_TWINS = (("oa-portal", "10.30.2.27"), ("oa-portal", "10.30.2.28"))
DEFAULT_LOOKS_LIKE = {"T9": "10.30.2.21"}
DEFAULT_LINK_PAIRS = {"L6": (("10.20.1.12", "10.20.1.112"),)}
DEFAULT_NEGATIVE = {"L6": ("10.20.1.114",)}
# accept deadlines after the first SUSPECT (gate 6), seconds
ACCEPT_WITHIN_S = {"L5": 3 * 86400.0, "L9": 4 * 86400.0}
NOISE_ROLES = frozenset({"", "none", "noise", "unique", "-1"})

AXES_VOCAB = frozenset({
    "volume", "shape", "peer", "temporal", "categorical", "breadth", "exfil", "sequence",
    "identity", "change", "c2", "discovery", "privilege", "credential", "collection",
    "lateral", "exfiltration", "off_hours", "behavior", "app", "dns", "tls", "timing",
    "transport", "probe", "comp"})
EVENT_KINDS = frozenset({
    "incident", "first_seen", "rare_access", "class_adopted", "client_change",
    "client_impersonation", "identity_mismatch", "unknown_identity", "low_identifiability",
    "entity_resolution", "possible_impersonation", "shared_ip", "identity_moved",
    "link_retracted", "new_entity_matched", "new_entity_unmatched", "class_transition",
    "class_split", "class_merge", "peer_outlier", "system_shift", "coherent_shift",
    "class_shift", "class_adoption_risky", "schedule_shift", "beacon", "budget_exceeded",
    "baseline_creep", "regime", "pipeline_degraded"})
# discrete event kind -> owner engine (engines.md), for 'B16'-style expectations
EVENT_OWNER = {
    "first_seen": "b08", "rare_access": "b08", "class_adopted": "b08",
    "client_change": "b09", "client_impersonation": "b09",
    "identity_mismatch": "b16", "unknown_identity": "b16", "low_identifiability": "b16",
    "entity_resolution": "b17", "possible_impersonation": "b17", "shared_ip": "b17",
    "identity_moved": "b17", "link_retracted": "b17",
    "new_entity_matched": "b02", "new_entity_unmatched": "b02", "class_transition": "b02",
    "class_split": "b02", "class_merge": "b02", "peer_outlier": "b02",
    "system_shift": "b05", "coherent_shift": "b05", "class_shift": "b18",
    "class_adoption_risky": "b18", "schedule_shift": "b07", "beacon": "b12",
    "budget_exceeded": "b13", "baseline_creep": "b14", "regime": "b28", "incident": "b27"}
KNOWN_NAMES = frozenset(DETECTORS) | AXES_VOCAB | EVENT_KINDS | frozenset(FAMILIES)

# Per-engine CPU budgets at 40 entities (architecture section 7), ms per tick.
ENGINE_BUDGET_MS = {"B01": 12, "B02": 1, "B03": 7, "B04": 6, "B05": 1, "B06": 5, "B07": 1,
                    "B08": 3, "B09": 1, "B10": 5, "B11-B13": 6, "B14": 3, "B15": 5,
                    "B16": 10, "B17": 1, "B18": 3, "P2": 6, "B23-B28": 9, "B30": 5}

TARGETS = {
    "loud_recall": 1.0, "subtle_recall": 0.90, "overall_recall": 0.95,
    "loud_ttd_ticks": 2, "loud_ttd_wall_s": 1800.0, "pack_e_slack_s": 900.0,
    "far_low": 0.2, "far_medium": 0.05, "far_high_total": 1, "far_critical": 0,
    "bursty_steady_ratio": 2.0, "cadence_ratio": (0.5, 2.0),
    "legit_ok_frac": 0.95, "legit_max_risk": 30.0,
    "notif_per_tp": 3.0, "incidents_per_episode": 1.0, "class_incidents_per_event": 1,
    "anchor_current_sigma": 0.3, "anchor_reference_sigma": 0.1,
    "ks_d": 0.05, "exceed_ratio": (0.5, 2.0), "evidence_cusum_rate": 0.045,
    "family_rate_mult": 2.0,
    "id_window_top1": 0.95, "id_tick_top1": 0.85, "id_eer": 0.05, "id_eer_frac": 0.90,
    "twin_eer": 0.2, "sep_spearman": 0.8, "t9b_frac": 0.90, "l6_recall": 0.95,
    "t19_frac": 0.90,
    "ari": 0.90, "noise_frac": 0.10, "sub_purity": 0.8, "refit_ari": 0.95,
    "l8_prob": 0.8, "t21_frac": 0.90,
    "hit3": 0.8, "cf_valid": 0.9, "nat_range": 1.0,
    "portrait_jaccard": 0.8, "template_recall": 0.8, "coverage": (0.85, 0.95),
    "class_portraits": 1.0,
    "feedback_cut": 0.5, "feedback_recall_drop": 0.02,
    "pack_seed_s": 360.0, "lib3_p95_ms": 80.0, "smoke_total_s": 30.0, "smoke_lib3_s": 12.0,
    "mem_per_entity": 12 * 1024 * 1024,
}


def pack_is(name: Any, letter: str) -> bool:
    """'A', 'Pack A', 'pack_a', 'PACK-A' all name Pack A (but 'smoke' is not E)."""
    n = str(name).upper().strip()
    if n.startswith("PACK"):
        n = n[4:]
    return n.strip(" _-") == letter.upper()


def sev_rank(s: Any) -> int:
    """Rank of a severity ('info' 0 .. 'critical' 4); unknown -> 0."""
    if s is None:
        return -1
    v = getattr(s, "value", s)
    return SEV_RANK.get(str(v).lower(), 0)


def base_id(sid: Any) -> str:
    """'T1', 'T6b', 'L10' from ids such as "T1'", 'T1_e', 'T6b-2'."""
    m = re.match(r"^\s*([TL]\d+b?)", str(sid))
    return m.group(1) if m else str(sid)


def _f(v: Any, default: float = math.nan) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _key(system: str, entity: Any) -> str:
    e = str(entity)
    return e if "|" in e else f"{system}|{e}"


def row_keys(row: Mapping[str, Any]) -> List[str]:
    s = str(row.get("system") or "")
    return [_key(s, e) for e in (row.get("entities") or [])]


def row_label(row: Mapping[str, Any]) -> str:
    return str(row.get("label") or "")


def is_class_scenario(row: Mapping[str, Any]) -> bool:
    return (row.get("level") == "class" or bool(row.get("class_scenario"))
            or base_id(row.get("scenario_id")) in CLASS_SCENARIOS)


def grace_s(dt: float) -> float:
    """TP window extension after t_end: max(4 ticks, 1 h)."""
    return max(4.0 * dt, 3600.0)


def deadline_s(row: Mapping[str, Any], dt: float) -> Optional[float]:
    """Scenario deadline in seconds. Accepts max_ttd_s, max_ttd_ticks, or
    max_ttd as seconds (number), {'s'|'ticks': n} or '2 ticks' / '6h' / '30min'."""
    if row.get("max_ttd_s") is not None:
        return _f(row["max_ttd_s"])
    if row.get("max_ttd_ticks") is not None:
        return _f(row["max_ttd_ticks"]) * dt
    v = row.get("max_ttd")
    if v is None:
        return None
    if isinstance(v, Mapping):
        if "s" in v:
            return _f(v["s"])
        if "ticks" in v:
            return _f(v["ticks"]) * dt
        return None
    if isinstance(v, str):
        m = re.match(r"^\s*([\d.]+)\s*([a-z]*)\s*$", v.lower())
        if not m:
            return None
        n, unit = float(m.group(1)), m.group(2)
        mult = {"": 1.0, "s": 1.0, "sec": 1.0, "min": 60.0, "m": 60.0, "h": 3600.0,
                "d": 86400.0, "tick": dt, "ticks": dt}.get(unit)
        return None if mult is None else n * mult
    return _f(v)


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def bootstrap_ci(values: Sequence[float], stat: Callable[[np.ndarray], float] = np.median,
                 n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI of `stat` over the given per-seed values."""
    x = np.asarray([v for v in values if v is not None and not _isnan(v)], dtype=float)
    if x.size == 0:
        return None, None
    if x.size == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.apply_along_axis(stat, 1, x[idx])
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def _isnan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _median(values: Iterable[Any]) -> Optional[float]:
    x = [float(v) for v in values if v is not None and not _isnan(v)]
    return float(np.median(x)) if x else None


def _quantile(values: Iterable[Any], q: float) -> Optional[float]:
    x = [float(v) for v in values if v is not None and not _isnan(v)]
    return float(np.quantile(x, q)) if x else None


def ks_uniform(p: np.ndarray) -> Tuple[Optional[float], int]:
    """KS D of p-values against U(0,1) (NaN dropped); (None, n) if n < 2."""
    x = np.sort(np.asarray(p, dtype=float))
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return None, int(n)
    x = np.clip(x, 0.0, 1.0)
    i = np.arange(1, n + 1)
    d = max(float(np.max(i / n - x)), float(np.max(x - (i - 1) / n)))
    return d, int(n)


def rising_edges(flags: np.ndarray) -> int:
    """Number of 0 -> 1 transitions (a run starting at index 0 counts)."""
    f = np.asarray(flags).astype(bool)
    if f.size == 0:
        return 0
    return int(f[0]) + int(np.sum(f[1:] & ~f[:-1]))


def _merge(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _overlap_len(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _in_any(t: float, intervals: Sequence[Tuple[float, float]]) -> bool:
    return any(a <= t <= b for a, b in intervals)


def _ticks_between(tick_ts: np.ndarray, t0: float, t1: float, dt: float) -> int:
    """Ticks with t0 <= ts <= t1 (the onset tick counts as 1)."""
    if tick_ts is not None and len(tick_ts):
        ts = np.asarray(tick_ts, dtype=float)
        return int(np.sum((ts >= t0 - 1e-6) & (ts <= t1 + 1e-6)))
    return int(math.floor((t1 - t0) / dt + 1e-9)) + 1


# --------------------------------------------------------------------------- #
# Range-based precision / recall (Tatbul et al. 2018)
# --------------------------------------------------------------------------- #
Range = Tuple[frozenset, float, float]      # (entity keys, t0, t1)


def _bias_cum(u: float, bias: str) -> float:
    """Integral of the positional weight w(x) over [0, u] (u in [0, 1])."""
    u = min(max(u, 0.0), 1.0)
    if bias == "flat":
        return u
    if bias == "front":
        return u - u * u / 2.0
    if bias == "back":
        return u * u / 2.0
    if bias == "middle":
        return u * u / 2.0 if u <= 0.5 else 0.125 + (u - 0.5) - (u * u - 0.25) / 2.0
    raise ValueError(f"unknown bias {bias!r}")


def _omega(r0: float, r1: float, o0: float, o1: float, bias: str) -> float:
    """Positional overlap reward of [o0, o1] inside the range [r0, r1]."""
    L = r1 - r0
    if L <= 0:
        return 1.0 if o1 >= o0 else 0.0
    tot = _bias_cum(1.0, bias)
    return (_bias_cum((o1 - r0) / L, bias) - _bias_cum((o0 - r0) / L, bias)) / tot


def _range_score(targets: Sequence[Range], others: Sequence[Range], alpha: float,
                 bias: str) -> Optional[float]:
    if not targets:
        return None
    vals = []
    for keys, r0, r1 in targets:
        ov = [(max(r0, p0), min(r1, p1)) for pk, p0, p1 in others
              if (pk & keys) and min(r1, p1) >= max(r0, p0) and (min(r1, p1) > max(r0, p0)
                                                                   or r1 == r0 or p1 == p0)]
        existence = 1.0 if ov else 0.0
        if ov:
            gamma = 1.0 if len(ov) == 1 else 1.0 / len(ov)
            reward = gamma * sum(_omega(r0, r1, a, b, bias) for a, b in ov)
        else:
            reward = 0.0
        vals.append(alpha * existence + (1 - alpha) * min(reward, 1.0))
    return float(np.mean(vals))


def range_prf(real: Sequence[Range], pred: Sequence[Range], alpha: float = 0.0,
              bias: str = "flat") -> Dict[str, Optional[float]]:
    """Range-based precision, recall and F1 (Tatbul 2018). Recall uses the
    existence weight `alpha` and positional `bias`; precision uses alpha = 0
    and flat bias, as recommended there. Ranges only overlap when their entity
    key sets intersect."""
    rec = _range_score(real, pred, alpha, bias)
    prec = _range_score(pred, real, 0.0, "flat")
    f1 = None
    if rec is not None and prec is not None:
        f1 = 0.0 if rec + prec == 0 else 2 * prec * rec / (prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1}


def point_adjusted_prf(real: Sequence[Range], pred: Sequence[Range]) -> Dict[str, Optional[float]]:
    """Point-adjusted scores in time units: a real range touched by any
    prediction counts as fully detected; predicted time outside every real
    range of the same entity is a false positive. Reference only."""
    tp = fn = 0.0
    for keys, r0, r1 in real:
        L = max(r1 - r0, 1e-9)
        hit = any((pk & keys) and min(r1, p1) >= max(r0, p0) for pk, p0, p1 in pred)
        if hit:
            tp += L
        else:
            fn += L
    fp = 0.0
    for pk, p0, p1 in pred:
        L = max(p1 - p0, 0.0)
        inside = _merge((max(p0, r0), min(p1, r1)) for rk, r0, r1 in real if rk & pk)
        fp += max(0.0, L - sum(b - a for a, b in inside))
    prec = tp / (tp + fp) if tp + fp > 0 else None
    rec = tp / (tp + fn) if tp + fn > 0 else None
    f1 = None
    if prec is not None and rec is not None:
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1}


# --------------------------------------------------------------------------- #
# RunView: one run, indexed for scoring
# --------------------------------------------------------------------------- #
def _strs(v: Any) -> List[str]:
    """Flatten a value into the strings it mentions (aliases, members, ...)."""
    out: List[str] = []
    if v is None:
        return out
    if isinstance(v, str):
        return [v]
    if isinstance(v, Mapping):
        for k in ("entity", "ip", "key", "to", "from", "src", "dst"):
            if k in v:
                out += _strs(v[k])
        for k in ("members", "entities", "chain", "ips", "aliases"):
            if k in v:
                out += _strs(v[k])
        return out
    if isinstance(v, (list, tuple, set, frozenset)):
        for x in v:
            out += _strs(x)
    return out


def _norm_feature(name: Any) -> str:
    s = str(name.get("feature") or name.get("name") or name.get("key") or "") \
        if isinstance(name, Mapping) else str(name)
    return s.split(".")[-1].strip().lower()


def split_episodes(incidents: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """One record per open episode of each incident.

    B27 reuses an incident's id when it reopens within 24 h (engines.md B27),
    so its `opened` stays at the FIRST opening. A reopening is a new opening
    (it notifies again), and a threat that reopens an incident closed hours
    before its onset is detected by that reopening: scored on the original
    `opened` it was invisible (eval pack A, 900-s warm-up: T5, T6b, T13 were
    all reopenings of FP incidents closed before the scenario started).
    Each episode keeps the id and gets `opened` = its (re)open time, the
    history points of that episode (through its closing point) and
    `episode` = (index, start, end) (end None while open)."""
    out: List[Dict[str, Any]] = []
    for inc in incidents:
        h = RunView.history(inc)
        if not inc.get("history") or len(h) < 2:
            out.append(dict(inc))
            continue
        eps: List[List[Dict[str, Any]]] = [[]]
        closed_prev = False
        for p in h:
            if closed_prev and p.get("status") != "closed":
                eps.append([])
            eps[-1].append(p)
            closed_prev = p.get("status") == "closed"
        if len(eps) == 1:
            out.append(dict(inc))
            continue
        for k, ep in enumerate(eps):
            d = dict(inc)
            start = float(inc.get("opened", ep[0]["ts"])) if k == 0 else float(ep[0]["ts"])
            ended = ep[-1].get("status") == "closed"
            d["opened"] = start
            d["history"] = ep
            d["status"] = "closed" if ended else inc.get("status", "open")
            d["severity"] = SEVERITIES[max(sev_rank(p.get("severity")) for p in ep)]
            d["episode"] = [k, start, float(ep[-1]["ts"]) if ended else None]
            out.append(d)
    return out


class RunView:
    """Indexes a RunResult (or an equivalent mapping) for the metric functions."""

    def __init__(self, run: Any) -> None:
        self.run = run
        self.pack = str(_get(run, "pack", ""))
        self.seed = int(_get(run, "seed", 0) or 0)
        self.dt = float(_get(run, "scenario_dt", 900.0) or 900.0)
        win = _get(run, "scenario_window", None) or (0.0, 0.0)
        self.t0, self.t1 = float(win[0]), float(win[1])
        self.tick_ts = np.asarray(_get(run, "tick_ts", None) if _get(run, "tick_ts", None)
                                  is not None else [], dtype=float)
        self.truth: List[Dict[str, Any]] = [dict(r) for r in (_get(run, "truth", None) or [])]
        self.incidents: List[Dict[str, Any]] = split_episodes(
            list(_get(run, "incidents", None) or []))
        self.events: List[Dict[str, Any]] = sorted(_get(run, "events", None) or [],
                                                   key=lambda e: float(e.get("ts", 0.0)))
        self.series: Dict[str, Dict[str, Any]] = dict(_get(run, "series", None) or {})
        self.profiles: Dict[str, Dict[str, Any]] = dict(_get(run, "profiles", None) or {})
        self.models: Dict[str, Any] = dict(_get(run, "models", None) or {})
        self.personas: Dict[str, Dict[str, Any]] = dict(_get(run, "personas", None) or {})
        self.first_seen: Dict[str, float] = dict(_get(run, "entity_first_seen", None) or {})
        self.fixtures: Dict[str, Any] = dict(_get(run, "fixtures", None) or {})
        self._index()

    # -- indexes -------------------------------------------------------------
    def _index(self) -> None:
        self.ev_by_key: Dict[str, List[Dict[str, Any]]] = {}
        self.ev_by_incident: Dict[str, List[Dict[str, Any]]] = {}
        for ev in self.events:
            self.ev_by_key.setdefault(_key(ev.get("system", ""), ev.get("entity", "")),
                                      []).append(ev)
            iid = ev.get("incident_id") or (ev.get("extra") or {}).get("incident_id")
            if iid:
                self.ev_by_incident.setdefault(str(iid), []).append(ev)
        # per-key ts arrays (events are ts-sorted) for O(log n) window slicing
        self.ev_ts: Dict[str, np.ndarray] = {
            k: np.fromiter((float(e.get("ts", 0.0)) for e in v), dtype=float, count=len(v))
            for k, v in self.ev_by_key.items()}
        # aliases: continuity + model.link links
        self.aliases: Dict[str, Set[str]] = {}
        for k, prof in self.profiles.items():
            cont = (prof.get("extra") or {}).get("continuity") or {}
            s = k.partition("|")[0]
            for fld in ("aliases", "linked_from", "linked_to"):
                for a in _strs(cont.get(fld)):
                    self._alias(k, _key(s, a))
        self.actors: List[Set[str]] = []
        for s, m in (self.models.get("link") or {}).items():
            if not isinstance(m, Mapping):
                continue
            links = m.get("links") or []
            for ln in (links.values() if isinstance(links, Mapping) else links):
                ips = [_key(s, x) for x in _strs(ln)]
                for a in ips:
                    for b in ips:
                        if a != b:
                            self._alias(a, b)
            actors = m.get("actors") or []
            for act in (actors.values() if isinstance(actors, Mapping) else actors):
                mem = {_key(s, x) for x in _strs(act)}
                if len(mem) >= 2:
                    self.actors.append(mem)
        # classes: final assignment plus every refit seen during the run
        self.class_of: Dict[str, Set[str]] = {}
        self.class_members: Dict[str, Set[str]] = {}
        snaps = list(self.models.get("class_history") or [])
        if self.models.get("class"):
            snaps.append(self.models["class"])
        for snap in snaps:
            for k, a in (snap.get("assign") or {}).items():
                for ck in self._class_keys_of(k, a):
                    self.class_of.setdefault(k, set()).add(ck)
                    self.class_members.setdefault(ck, set()).add(k)
        # entities
        self.real: Set[str] = set(self.personas) | set(self.first_seen)
        self.real |= {k for k in self.series if "|class:" not in k and "|__" not in k}
        self.row_exp: List[Set[str]] = [self.expand(r) for r in self.truth]
        scen: Set[str] = set()
        for r, exp in zip(self.truth, self.row_exp):
            scen |= {k for k in exp if "|class:" not in k}
            scen |= set(row_keys(r))
        self.scenario_keys = scen
        self.control: Set[str] = {k for k in self.real if k not in scen}
        self.exclusions: Dict[str, List[Tuple[float, float]]] = {k: [] for k in self.control}
        for r, exp in zip(self.truth, self.row_exp):
            t0, t1 = _f(r.get("t_start")), _f(r.get("t_end"))
            if math.isnan(t0):
                continue
            t1 = t0 if math.isnan(t1) else t1
            win = (t0 - 3600.0, t1 + 86400.0)
            cks = {c for k in row_keys(r) for c in self.class_of.get(k, ())}
            cks |= {c for c in exp if "|class:" in c}
            cks |= {_key(str(r.get("system") or ""), c) for c in _strs(r.get("classes"))}
            system_wide = row_label(r) == "system_change" or r.get("level") == "system"
            for e in self.control:
                if (is_class_scenario(r) and self.class_of.get(e, set()) & cks) or \
                        (system_wide and e.partition("|")[0] == str(r.get("system"))):
                    self.exclusions[e].append(win)
        for e in self.exclusions:
            self.exclusions[e] = _merge(self.exclusions[e])

    def _alias(self, a: str, b: str) -> None:
        if a != b:
            self.aliases.setdefault(a, set()).add(b)
            self.aliases.setdefault(b, set()).add(a)

    @staticmethod
    def _class_keys_of(k: str, a: Mapping[str, Any]) -> List[str]:
        s = k.partition("|")[0]
        out = []
        role = a.get("role")
        if role is not None and str(role).lower() not in NOISE_ROLES:
            out.append(f"{s}|class:{role}")
        for st in a.get("static") or []:
            out.append(f"{s}|class:static:{st}")
        if a.get("pool"):
            out.append(f"{s}|class:pool:{a['pool']}")
        return out

    def expand(self, row: Mapping[str, Any]) -> Set[str]:
        """Scenario entity keys + continuity aliases + actor chain (+ class
        keys for class scenarios)."""
        keys = set(row_keys(row))
        exp = set(keys)
        for k in keys:
            exp |= self.aliases.get(k, set())
        for act in self.actors:
            if act & exp:
                exp |= act
        if is_class_scenario(row):
            for k in keys:
                exp |= self.class_of.get(k, set())
            s = str(row.get("system") or "")
            exp |= {_key(s, c) for c in _strs(row.get("classes"))}
        return exp

    # -- incidents -------------------------------------------------------------
    @staticmethod
    def inc_keys(inc: Mapping[str, Any]) -> Set[str]:
        s = str(inc.get("system", ""))
        return {_key(s, inc.get("entity", ""))} | {_key(s, x) for x in inc.get("entities") or []}

    @staticmethod
    def history(inc: Mapping[str, Any]) -> List[Dict[str, Any]]:
        h = list(inc.get("history") or [])
        if h:
            return sorted(h, key=lambda p: float(p.get("ts", 0.0)))
        return [{"ts": float(inc.get("opened", 0.0)), "severity": inc.get("severity", "low"),
                 "status": inc.get("status", "open"), "axes": inc.get("axes") or [],
                 "kinds": inc.get("kinds") or []}]

    @staticmethod
    def max_severity(inc: Mapping[str, Any]) -> int:
        return max([sev_rank(p.get("severity")) for p in RunView.history(inc)]
                   + [sev_rank(inc.get("severity"))])

    @staticmethod
    def _episode_span(inc: Mapping[str, Any]) -> Tuple[float, float]:
        ep = inc.get("episode")
        if not ep:
            return -math.inf, math.inf
        return float(ep[1]), (math.inf if ep[2] is None else float(ep[2]))

    def close_ts(self, inc: Mapping[str, Any]) -> Optional[float]:
        for p in self.history(inc):
            if p.get("status") == "closed":
                return float(p["ts"])
        if inc.get("status") == "closed":
            return float(inc.get("last_seen") or inc.get("opened") or 0.0)
        return None

    def incident_detectors(self, inc: Mapping[str, Any], until: float = math.inf) -> Set[str]:
        """Detectors / kinds an incident carries through p_by_detector (in
        evidence, explanation or its linked events) up to `until`."""
        out: Set[str] = set()
        since, _ = self._episode_span(inc)        # this (re)opening's evidence only
        expl = inc.get("explanation") or {}
        for src in (expl.get("p_by_detector"), inc.get("p_by_detector")):
            if isinstance(src, Mapping):
                out |= {str(k).lower() for k in src}
        for ev in inc.get("evidence") or []:
            if not isinstance(ev, Mapping):
                continue
            if not since <= _f(ev.get("ts"), since) <= until:
                continue
            pbd = ev.get("p_by_detector")
            if isinstance(pbd, Mapping):
                out |= {str(k).lower() for k in pbd}
            for fld in ("detector", "detectors", "kind", "source"):
                out |= {x.lower() for x in _strs(ev.get(fld))}
            out |= {str(a).lower() for a in ev.get("axes") or []}
        for ev in self.ev_by_incident.get(str(inc.get("id", "")), []):
            if since <= float(ev.get("ts", 0.0)) <= until:
                out |= {str(k).lower() for k in (ev.get("p_by_detector") or {})}
                out |= {str(a).lower() for a in ev.get("axes") or []}
        return out

    def event_hits(self, keys: Set[str], t0: float, t1: float) -> Set[str]:
        out: Set[str] = set()
        for k in keys:
            evs = self.ev_by_key.get(k)
            if not evs:
                continue
            ts = self.ev_ts[k]
            i0, i1 = np.searchsorted(ts, t0, "left"), np.searchsorted(ts, t1, "right")
            for ev in evs[i0:i1]:
                if ev.get("kind") != "incident":
                    out.add(str(ev.get("kind")).lower())
                out |= {str(a).lower() for a in ev.get("axes") or []}
                out |= {str(d).lower() for d in (ev.get("p_by_detector") or {})}
                ex = ev.get("extra") or {}
                out |= {x.lower() for x in _strs(ex.get("detector"))}
                out |= {x.lower() for x in _strs(ex.get("detectors"))}
        return out

    def notifications(self, inc: Mapping[str, Any]) -> int:
        lo, hi = self._episode_span(inc)
        evs = [e for e in self.ev_by_incident.get(str(inc.get("id", "")), [])
               if e.get("kind") == "incident" and lo <= float(e.get("ts", 0.0)) <= hi]
        if evs:
            return sum(1 for e in evs
                       if str((e.get("extra") or {}).get("state", "")).lower()
                       in ("open", "escalate"))
        # no incident events recorded: the open plus every escalation
        n, best = 0, -1
        for p in self.history(inc):
            r = sev_rank(p.get("severity"))
            if r > best:
                n, best = n + 1, r
        return n


# --------------------------------------------------------------------------- #
# Expected detector / axis matching
# --------------------------------------------------------------------------- #
def _tokens(item: Any) -> Tuple[Set[str], Set[str]]:
    """(names, engine codes) mentioned by one expected_detectors/axes item,
    e.g. 'B04 marg_int (NB lower tail)' -> ({'marg_int'}, {'b04'})."""
    toks = [t for t in re.split(r"[^a-z0-9_]+", str(item).lower()) if t]
    codes = {t for t in toks if re.fullmatch(r"[bdr]\d{1,2}", t)}
    names = {t for t in toks if t in KNOWN_NAMES or ("_" in t and t not in codes)}
    return names, codes


def _hit_codes(hits: Set[str]) -> Set[str]:
    out = set()
    for h in hits:
        if h in DETECTOR_INFO:
            out.add(str(DETECTOR_INFO[h]["owner"]).lower())
        if h in EVENT_OWNER:
            out.add(EVENT_OWNER[h])
    return out


def expected_match(row: Mapping[str, Any], hits: Set[str]) -> Tuple[bool, List[str]]:
    """True if `hits` include at least one expected detector or axis. An
    item that names detectors/axes/kinds matches on those names; an item
    that only names an engine code matches anything that engine owns. With
    nothing expected, any incident qualifies."""
    items = list(row.get("expected_detectors") or []) + list(row.get("expected_axes") or [])
    if not items:
        return True, []
    codes_hit = _hit_codes(hits)
    matched = []
    for it in items:
        names, codes = _tokens(it)
        if names:
            if names & hits:
                matched.append(str(it))
        elif codes and codes & codes_hit:
            matched.append(str(it))
    return bool(matched), matched


# --------------------------------------------------------------------------- #
# Per-scenario outcomes (gates 1, 2, 4, 5)
# --------------------------------------------------------------------------- #
def _detect_on(view: RunView, row: Mapping[str, Any], keys: Set[str]) -> Dict[str, Any]:
    t_start = _f(row.get("t_start"))
    t_end = _f(row.get("t_end"), t_start)
    lo, hi = t_start, t_end + grace_s(view.dt)
    req = sev_rank(row.get("required_severity") or "low")
    best: Optional[Tuple[float, Dict[str, Any], List[str]]] = None
    cands = []
    for inc in view.incidents:
        if not (view.inc_keys(inc) & keys):
            continue
        opened = float(inc.get("opened", 0.0))
        if opened < lo - 1e-6 or opened > hi + 1e-6:
            continue
        cands.append(inc)
        for pt in view.history(inc):
            ts = float(pt.get("ts", opened))
            if ts > hi + 1e-6:
                break
            if sev_rank(pt.get("severity")) < req:
                continue
            hits = {str(a).lower() for a in (pt.get("axes") or inc.get("axes") or [])}
            hits |= {str(k).lower() for k in (pt.get("kinds") or inc.get("kinds") or [])}
            hits |= view.incident_detectors(inc, until=ts)
            hits |= view.event_hits(keys, t_start - view.dt, ts)
            ok, matched = expected_match(row, hits)
            if ok:
                if best is None or ts < best[0]:
                    best = (ts, inc, matched)
                break
    out: Dict[str, Any] = {
        "detected": best is not None,
        "n_incidents": len({str(i.get("id", id(i))) for i in cands}),   # reopenings: once
        "max_severity": (SEVERITIES[max(view.max_severity(i) for i in cands)] if cands else None),
        "incident_ids": sorted({str(i.get("id")) for i in cands}),
    }
    if best is not None:
        ts, inc, matched = best
        out.update({"t_detect": ts, "ttd_s": ts - t_start,
                    "ttd_ticks": _ticks_between(view.tick_ts, t_start, ts, view.dt),
                    "tp_incident": str(inc.get("id")), "matched": matched,
                    "notifications": view.notifications(inc)})
    return out


def scenario_outcome(view: RunView, idx: int) -> Dict[str, Any]:
    """Detection outcome of malicious truth row `idx`."""
    row = view.truth[idx]
    sid = str(row.get("scenario_id"))
    base = base_id(sid)
    exp = view.row_exp[idx]
    keys = row_keys(row)
    mode = row.get("match") or ("all" if len(keys) > 1 and base not in CLASS_SCENARIOS
                                and base not in ACTOR_SCENARIOS else "any")
    if mode == "all":
        parts = []
        for k in keys:
            sub = view.expand({**row, "entities": [k]})
            parts.append(_detect_on(view, row, sub))
        detected = all(p["detected"] for p in parts)
        res = {"detected": detected,
               "n_incidents": sum(p["n_incidents"] for p in parts),
               "incident_ids": [i for p in parts for i in p["incident_ids"]],
               "max_severity": max((p["max_severity"] for p in parts if p["max_severity"]),
                                   key=sev_rank, default=None),
               "per_entity": {k: p["detected"] for k, p in zip(keys, parts)}}
        if detected:
            last = max(parts, key=lambda p: p["t_detect"])
            res.update({k: last[k] for k in ("t_detect", "ttd_s", "ttd_ticks", "tp_incident",
                                             "matched")})
            res["notifications"] = float(np.mean([p["notifications"] for p in parts]))
            res["tp_incidents"] = [p["tp_incident"] for p in parts]
    else:
        res = _detect_on(view, row, exp)
        if res["detected"]:
            res["tp_incidents"] = [res["tp_incident"]]
    dl = deadline_s(row, view.dt)
    loud = base in LOUD if row.get("loudness") is None else row.get("loudness") == "loud"
    within = res["detected"]
    if within and dl is not None:
        within = res["ttd_s"] <= dl + 1e-6
    if within and loud:
        # loud: <= 2 ticks at 900 s, <= 30 min wall at any other cadence
        within = (res["ttd_ticks"] <= TARGETS["loud_ttd_ticks"] if abs(view.dt - 900.0) < 1
                  else res["ttd_s"] <= TARGETS["loud_ttd_wall_s"])
    episodes = max(1, len(keys) if mode == "all" else 1)
    res.update({
        "scenario_id": sid, "base": base, "pack": view.pack, "seed": view.seed,
        "label": row_label(row), "system": row.get("system"), "entities": keys,
        "t_start": _f(row.get("t_start")), "t_end": _f(row.get("t_end")),
        "required_severity": row.get("required_severity") or "low",
        "deadline_s": dl, "within_deadline": bool(within), "dt": view.dt,
        "loudness": "loud" if loud else ("p2" if base in P2_SCENARIOS else "subtle"),
        "incidents_per_episode": res["n_incidents"] / episodes,
    })
    return res


def legit_outcome(view: RunView, idx: int) -> Dict[str, Any]:
    """Compliance of a legitimate / system-change truth row (gate 4)."""
    row = view.truth[idx]
    exp = view.row_exp[idx]
    t0 = _f(row.get("t_start"))
    t1 = _f(row.get("t_end"), t0)
    lo, hi = t0 - 3600.0, t1 + 86400.0
    incs = [i for i in view.incidents if view.inc_keys(i) & exp
            and lo <= float(i.get("opened", 0.0)) <= hi]
    max_sev = max((view.max_severity(i) for i in incs), default=-1)
    allowed = sev_rank(row.get("max_allowed_severity") or "info")
    # an INFO allowance means "no incident at all": incidents start at LOW
    sev_ok = max_sev <= allowed if max_sev >= 0 else True
    risks = []
    for k in exp:
        ser = view.series.get(k)
        if ser is None:
            continue
        ts = np.asarray(ser.get("ts", []), dtype=float)
        r = np.asarray(ser.get("risk", []), dtype=float)
        m = (ts >= lo) & (ts <= hi) & np.isfinite(r)
        if m.any():
            risks.append(float(np.max(r[m])))
    max_risk = max(risks) if risks else None
    risk_ok = max_risk is None or max_risk < TARGETS["legit_max_risk"]
    class_incs = [i for i in incs if str(i.get("entity", "")).startswith("class:")]
    # class-wide changes may allow a class incident while members stay quiet
    # (L1: 0 member incidents >= LOW; L10: 0 entity incidents >= LOW)
    member_max = max((view.max_severity(i) for i in incs if i not in class_incs), default=-1)
    member_allowed = row.get("max_member_severity")
    member_ok = (member_allowed is None or member_max < 0
                 or member_max <= sev_rank(member_allowed))
    return {
        "scenario_id": str(row.get("scenario_id")), "base": base_id(row.get("scenario_id")),
        "pack": view.pack, "seed": view.seed, "label": row_label(row),
        "entities": row_keys(row), "t_start": t0, "t_end": t1,
        "max_severity": SEVERITIES[max_sev] if max_sev >= 0 else None,
        "allowed": row.get("max_allowed_severity") or "info",
        "n_incidents": len({str(i.get("id", id(i))) for i in incs}),
        "n_class_incidents": len({str(i.get("id", id(i))) for i in class_incs}),
        "class_max_severity": (SEVERITIES[max(view.max_severity(i) for i in class_incs)]
                               if class_incs else None),
        "member_max_severity": SEVERITIES[member_max] if member_max >= 0 else None,
        "max_risk": max_risk, "sev_ok": bool(sev_ok), "risk_ok": bool(risk_ok),
        "member_ok": bool(member_ok), "ok": bool(sev_ok and risk_ok and member_ok),
    }


# --------------------------------------------------------------------------- #
# False alarms on control entities (gate 3)
# --------------------------------------------------------------------------- #
def _group(view: RunView, key: str) -> str:
    arch = str((view.personas.get(key) or {}).get("archetype", "")).lower()
    if not arch:
        return "unknown"
    return "bursty" if arch in HUMAN_ARCHETYPES else "steady"


def entity_days(view: RunView, key: str) -> float:
    start = max(view.t0, view.first_seen.get(key, view.t0))
    if view.t1 <= start:
        return 0.0
    excl = sum(_overlap_len(start, view.t1, a, b) for a, b in view.exclusions.get(key, []))
    return max(0.0, view.t1 - start - excl) / 86400.0


def far_counts(view: RunView) -> Dict[str, Any]:
    """Incidents on control entities, outside their exclusion windows, per
    severity threshold, with the clean entity-day exposure."""
    days = {k: entity_days(view, k) for k in view.control}
    counts = {s: 0 for s in ("low", "medium", "high", "critical")}
    groups: Dict[str, Dict[str, float]] = {}
    fps = []
    # an incident counts once however often it reopened (B27 reuses the id
    # within 24 h): the gate counts incidents, the reopenings are reported
    # as n_reopened; its severity is the max over the counted episodes
    per_id: Dict[str, Tuple[Mapping[str, Any], str, float, int]] = {}
    n_reopened = 0
    for inc in view.incidents:
        k = _key(str(inc.get("system", "")), inc.get("entity", ""))
        if k not in view.control:
            continue
        opened = float(inc.get("opened", 0.0))
        if opened < view.t0 or opened > view.t1 or _in_any(opened, view.exclusions.get(k, [])):
            continue
        iid = str(inc.get("id", id(inc)))
        r = view.max_severity(inc)
        if iid in per_id:
            n_reopened += 1
            first = per_id[iid]
            per_id[iid] = (first[0], first[1], first[2], max(first[3], r))
        else:
            per_id[iid] = (inc, k, opened, r)
    for inc, k, opened, r in per_id.values():
        g = groups.setdefault(_group(view, k), {"n_low": 0, "days": 0.0})
        for s in counts:
            if r >= SEV_RANK[s]:
                counts[s] += 1
        if r >= SEV_RANK["low"]:
            g["n_low"] += 1
        fps.append({"id": inc.get("id"), "entity": k, "opened": opened,
                    "severity": SEVERITIES[r], "axes": inc.get("axes") or []})
    for k, d in days.items():
        groups.setdefault(_group(view, k), {"n_low": 0, "days": 0.0})["days"] += d
    total = float(sum(days.values()))
    return {
        "entity_days": total, "n_control": len(view.control), "dt": view.dt,
        "n_low": counts["low"], "n_medium": counts["medium"], "n_high": counts["high"],
        "n_reopened": n_reopened,
        "n_critical": counts["critical"],
        "far_low": counts["low"] / total if total > 0 else None,
        "far_medium": counts["medium"] / total if total > 0 else None,
        "groups": groups, "false_alarms": fps[:200],
    }


# --------------------------------------------------------------------------- #
# Range scores (gate 1 report)
# --------------------------------------------------------------------------- #
def range_scores(view: RunView) -> Dict[str, Any]:
    real: List[Range] = []
    for row, exp in zip(view.truth, view.row_exp):
        if row_label(row) != "malicious":
            continue
        t0 = _f(row.get("t_start"))
        if math.isnan(t0):
            continue
        t1 = _f(row.get("t_end"), t0)
        real.append((frozenset(exp), t0, t1 + grace_s(view.dt)))
    pred: List[Range] = []
    for inc in view.incidents:
        if view.max_severity(inc) < SEV_RANK["low"]:
            continue
        o = float(inc.get("opened", 0.0))
        if o < view.t0 or o > view.t1:
            continue
        c = view.close_ts(inc)
        e = c if c is not None else float(inc.get("last_seen") or o)
        pred.append((frozenset(view.inc_keys(inc)), o, max(e, o + view.dt)))
    return {"range": range_prf(real, pred), "point_adjusted": point_adjusted_prf(real, pred),
            "n_real": len(real), "n_pred": len(pred)}


# --------------------------------------------------------------------------- #
# Calibration (gate 7)
# --------------------------------------------------------------------------- #
def cadence_class(dt: float) -> int:
    classes = (60, 300, 900, 3600)
    return min(classes, key=lambda c: abs(math.log(dt / c)))


def _clean_mask(view: RunView, key: str, ts: np.ndarray) -> np.ndarray:
    m = (ts >= view.t0) & (ts <= view.t1)
    for a, b in view.exclusions.get(key, []):
        m &= ~((ts >= a) & (ts <= b))
    return m


def calibration_stats(view: RunView) -> Dict[str, Any]:
    pooled: Dict[int, List[np.ndarray]] = {}
    e_all: List[np.ndarray] = []
    ev_alarms = st_alarms = 0
    path_edges: Dict[str, int] = {}
    ent_days = 0.0
    for k in view.control:
        ser = view.series.get(k)
        if ser is None:
            continue
        ts = np.asarray(ser.get("ts", []), dtype=float)
        if ts.size == 0:
            continue
        m = _clean_mask(view, k, ts)
        ent_days += entity_days(view, k)
        P = np.asarray(ser.get("p"), dtype=float).reshape(ts.size, -1)
        for j in range(P.shape[1]):
            pooled.setdefault(j, []).append(P[m, j])
        e = np.asarray(ser.get("e_day", []), dtype=float)
        if e.size == ts.size:
            e_all.append(e[m])
        paths = np.asarray([str(x) for x in ser.get("alarm_path", [])] or [""] * ts.size)
        if paths.size == ts.size:
            low = np.char.lower(paths)
            is_ev = (np.char.find(low, "cusum") >= 0) | (np.char.find(low, "evid") >= 0)
            ev_alarms += rising_edges(m & is_ev)
            st_alarms += rising_edges(m & (np.char.find(low, "single") >= 0))
        A = np.asarray(ser.get("acc_alarm"), dtype=float).reshape(ts.size, -1) \
            if len(ser.get("acc_alarm", [])) else np.zeros((ts.size, 0))
        for j in range(A.shape[1]):
            if j >= len(DETECTORS):
                break
            d = DETECTORS[j]
            info = DETECTOR_INFO.get(d, {})
            if info.get("kind") != "acc" or info.get("level") == "class":
                continue
            n = rising_edges(m & (A[:, j] > 0))
            path_edges[str(info.get("budget_path"))] = path_edges.get(
                str(info.get("budget_path")), 0) + n
    ks: Dict[str, Dict[str, Any]] = {}
    for j, chunks in pooled.items():
        x = np.concatenate(chunks) if chunks else np.empty(0)
        d, n = ks_uniform(x)
        name = DETECTORS[j] if j < len(DETECTORS) else str(j)
        if n:
            ks[name] = {"D": d, "n": n}
    e = np.concatenate(e_all) if e_all else np.empty(0)
    e = e[np.isfinite(e)]
    exceed = {}
    for x in (0.03, 3e-3):
        exceed[str(x)] = {"n": int(e.size), "k": int(np.sum(e <= x)),
                          "expected": float(e.size * x * view.dt / 86400.0)}
    # ACAT masking: family p > 0.5 while a member p < 1e-4
    acat_ticks = acat_incident_ticks = 0
    inc_ts: Dict[str, Set[float]] = {}
    for inc in view.incidents:
        for p in view.history(inc):
            for k in view.inc_keys(inc):
                inc_ts.setdefault(k, set()).add(float(p.get("ts", 0.0)))
    fam_idx = {f: [DETECTORS.index(d) for d in FAMILY_MEMBERS.get(f, [])] for f in FAMILIES}
    for k, ser in view.series.items():
        ts = np.asarray(ser.get("ts", []), dtype=float)
        if ts.size == 0 or not len(ser.get("p_family", [])):
            continue
        P = np.asarray(ser["p"], dtype=float).reshape(ts.size, -1)
        F = np.asarray(ser["p_family"], dtype=float).reshape(ts.size, -1)
        bad = np.zeros(ts.size, dtype=bool)
        for fi, f in enumerate(FAMILIES):
            if fi >= F.shape[1] or not fam_idx[f]:
                continue
            with np.errstate(all="ignore"):
                sub = P[:, fam_idx[f]]
                minp = np.where(np.all(np.isnan(sub), axis=1), np.nan,
                                np.nanmin(np.where(np.isnan(sub), np.inf, sub), axis=1))
            bad |= (F[:, fi] > 0.5) & (minp < 1e-4)
        acat_ticks += int(bad.sum())
        its = inc_ts.get(k, set())
        acat_incident_ticks += int(sum(1 for t in ts[bad] if float(t) in its))
    return {"ks": ks, "exceedance": exceed, "cc": cadence_class(view.dt), "dt": view.dt,
            "entity_days": ent_days, "evidence_cusum_alarms": ev_alarms,
            "single_tick_alarms": st_alarms, "acc_path_alarms": path_edges,
            "acat_ticks": acat_ticks, "acat_incident_ticks": acat_incident_ticks}


# --------------------------------------------------------------------------- #
# Poisoning / reversibility (gate 6)
# --------------------------------------------------------------------------- #
def _anchor_vec(model: Any, anchor: str, feature: str) -> Tuple[Optional[float], Optional[float]]:
    """(predictive mean, sigma15) of `feature` in the given anchor. Uses the
    B03 accessor module when present (lib/m_baseline.py), else a generic
    read of {'<anchor>': {'mean': {feature: v} | [F] | [B][F], 'sd15': ...}}."""
    try:  # accessor owned by B03 (helpers_api.md: consumers never read internals)
        from ..engines.behavior.lib import m_baseline  # type: ignore
        fn = getattr(m_baseline, "anchor_summary", None)
        if fn is not None:
            mu, sd = fn(model, anchor, feature)
            return _f(mu, None), _f(sd, None)  # type: ignore[arg-type]
    except Exception:
        pass
    if not isinstance(model, Mapping):
        return None, None
    a = model.get(anchor)
    if not isinstance(a, Mapping):
        return None, None

    def pick(v: Any) -> Optional[float]:
        if isinstance(v, Mapping):
            return _f(v.get(feature), None)  # type: ignore[arg-type]
        if isinstance(v, (list, tuple)) and v:
            try:
                from ..engines.behavior.lib.features import FEATURE_INDEX
                i = FEATURE_INDEX.get(feature)
            except Exception:  # pragma: no cover
                i = None
            if i is None:
                return None
            arr = np.asarray(v, dtype=float)
            if arr.ndim == 1 and i < arr.size:
                return float(arr[i])
            if arr.ndim == 2 and i < arr.shape[1]:
                return float(np.nanmean(arr[:, i]))
        return None

    return pick(a.get("mean")), pick(a.get("sd15", a.get("sigma15")))


def _nested_equal(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_nested_equal(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        try:
            x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            return x.shape == y.shape and bool(np.allclose(x, y, atol=tol, equal_nan=True))
        except (TypeError, ValueError):
            return len(a) == len(b) and all(_nested_equal(p, q, tol) for p, q in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return (_isnan(a) and _isnan(b)) or abs(_f(a) - _f(b)) <= tol
    return a == b


def poisoning_checks(view: RunView, outcomes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    snaps = view.models.get("baseline_snaps") or {}
    ctrl = view.models.get("control_history") or {}
    by_sid = {o["scenario_id"]: o for o in outcomes}
    rows = []
    for row, exp in zip(view.truth, view.row_exp):
        if row_label(row) != "malicious":
            continue
        sid = str(row.get("scenario_id"))
        base = base_id(sid)
        t0, t1 = _f(row.get("t_start")), _f(row.get("t_end"))
        t1 = t0 if math.isnan(t1) else t1
        rec: Dict[str, Any] = {"scenario_id": sid, "base": base}
        # control version bumps inside the malicious window
        bumps = 0
        for k in exp:
            prev = None
            for h in ctrl.get(k, []):
                v = h.get("version", h.get("store_version"))
                if prev is not None and v != prev and t0 <= float(h["ts"]) <= t1:
                    bumps += 1
                prev = v
        rec["control_bumps"] = bumps
        sn = snaps.get(sid) or {}
        pre, post = sn.get("pre"), sn.get("post")
        feats = [_norm_feature(f) for f in row.get("perturbed_features") or []]
        if pre and post and not post.get("truncated"):
            cur_d, ref_d, golden_ok = [], [], []
            for k, pm in (pre.get("models") or {}).items():
                qm = (post.get("models") or {}).get(k)
                if qm is None:
                    continue
                a, b = pm.get("model"), qm.get("model")
                if isinstance(a, Mapping) and isinstance(b, Mapping) and \
                        a.get("golden") is not None:
                    golden_ok.append(_nested_equal(a.get("golden"), b.get("golden")))
                for f in feats:
                    for anchor, bucket in (("current", cur_d), ("reference", ref_d)):
                        m0, s0 = _anchor_vec(a, anchor, f)
                        m1, _ = _anchor_vec(b, anchor, f)
                        if m0 is None or m1 is None:
                            continue
                        if anchor == "current" and base == "T2":
                            bucket.append(abs(m1 - m0))     # T2: 0.3 in log1p units
                        elif s0:
                            bucket.append(abs(m1 - m0) / s0)
            lim_cur = TARGETS["anchor_current_sigma"]
            rec["current_shift"] = max(cur_d) if cur_d else None
            rec["reference_shift"] = max(ref_d) if ref_d else None
            rec["current_ok"] = None if not cur_d else max(cur_d) <= lim_cur + 1e-9
            rec["reference_ok"] = (None if not ref_d
                                   else max(ref_d) <= TARGETS["anchor_reference_sigma"] + 1e-9)
            rec["golden_ok"] = all(golden_ok) if golden_ok else None
        # incident closes within max(8 ticks, 2 h) after the attack ends
        o = by_sid.get(sid) or {}
        close_by = t1 + max(8 * view.dt, 7200.0)
        if o.get("tp_incidents") and view.t1 >= close_by:
            ok = True
            for iid in o["tp_incidents"]:
                inc = next((i for i in view.incidents if str(i.get("id")) == iid), None)
                c = view.close_ts(inc) if inc else None
                ok &= c is not None and c <= close_by + 1e-6
            rec["closed_ok"] = ok
        # null KS within 96 ticks after the close deadline
        xs = []
        for k in row_keys(row):
            ser = view.series.get(k)
            if ser is None:
                continue
            ts = np.asarray(ser.get("ts", []), dtype=float)
            m = (ts > close_by) & (ts <= close_by + 96 * view.dt)
            if m.any():
                P = np.asarray(ser["p"], dtype=float).reshape(ts.size, -1)
                xs.append(P[m].ravel())
        if xs:
            d, n = ks_uniform(np.concatenate(xs))
            if d is not None and n >= 50:
                rec["null_ks"] = d
        rows.append(rec)
    # legitimate accepts and long DRIFTING without a label-queue entry
    accepts = []
    for row, exp in zip(view.truth, view.row_exp):
        if row_label(row) == "malicious":
            continue
        base = base_id(row.get("scenario_id"))
        limit = _f(row.get("accept_within_s"), ACCEPT_WITHIN_S.get(base, math.nan))
        if math.isnan(limit):
            continue
        sus = acc = None
        for k in exp:
            for ev in view.ev_by_key.get(k, []):
                if ev.get("kind") != "regime" or float(ev["ts"]) < _f(row.get("t_start")):
                    continue
                st = str((ev.get("extra") or {}).get("state", "")).lower()
                if st == "suspect" and sus is None:
                    sus = float(ev["ts"])
                if st == "accepted" and acc is None:
                    acc = float(ev["ts"])
        ok = None if sus is None else (acc is not None and acc - sus <= limit)
        accepts.append({"scenario_id": row.get("scenario_id"), "suspect": sus,
                        "accepted": acc, "limit_s": limit, "ok": ok})
    queue = {x for x in _strs((view.models.get("feedback") or {}).get("queue"))} \
        if isinstance(view.models.get("feedback"), Mapping) else set()
    long_drift = []
    for k, evs in view.ev_by_key.items():
        start = None
        for ev in evs:
            if ev.get("kind") != "regime":
                continue
            st = str((ev.get("extra") or {}).get("state", "")).lower()
            if st == "drifting" and start is None:
                start = float(ev["ts"])
            elif st in ("returned", "accepted", "rejected") and start is not None:
                if float(ev["ts"]) - start > 14 * 86400 and k not in queue \
                        and k.partition("|")[2] not in queue:
                    long_drift.append(k)
                start = None
        if start is not None and view.t1 - start > 14 * 86400 and k not in queue \
                and k.partition("|")[2] not in queue:
            long_drift.append(k)
    return {"scenarios": rows, "accepts": accepts, "long_drift": long_drift}


# --------------------------------------------------------------------------- #
# Identification (gate 8) and classes (gate 9)
# --------------------------------------------------------------------------- #
def _identity(view: RunView, key: str) -> Mapping[str, Any]:
    return ((view.profiles.get(key) or {}).get("extra") or {}).get("identity") or {}


def _twins(view: RunView) -> List[str]:
    tw = view.fixtures.get("twins")
    if tw:
        return [_key(t[0], t[1]) if isinstance(t, (list, tuple)) else str(t) for t in tw]
    return [f"{s}|{e}" for s, e in DEFAULT_TWINS]


def _mentions(obj: Any, target: str) -> bool:
    ip = target.partition("|")[2] or target
    return any(x == target or x == ip or x.endswith("|" + ip) for x in _strs(obj)) or (
        isinstance(obj, Mapping) and any(_mentions(v, target) for v in obj.values()
                                         if isinstance(v, (Mapping, list, tuple, str))))


def identification_stats(view: RunView) -> Dict[str, Any]:
    twins = _twins(view)
    indiv = [k for k, p in view.personas.items()
             if str(p.get("archetype", "")).lower() == "interactive" and k not in twins
             and k not in view.scenario_keys]
    win, tick, eer, sep, rec1 = [], [], [], [], []
    for k in indiv:
        idn = _identity(view, k)
        if not idn:
            continue
        w = _f(idn.get("recall1", idn.get("top1_window")), None)  # type: ignore[arg-type]
        t = _f(idn.get("recall1_tick", idn.get("top1_tick")), None)  # type: ignore[arg-type]
        e = _f(idn.get("eer_hard"), None)  # type: ignore[arg-type]
        if w is not None:
            win.append(w)
        if t is not None:
            tick.append(t)
        if e is not None:
            eer.append(e)
        s = _f((view.profiles.get(k) or {}).get("separability"), None)  # type: ignore[arg-type]
        if s is not None and w is not None:
            sep.append(s)
            rec1.append(w)
    spearman = None
    if len(sep) >= 4 and len(set(sep)) > 1 and len(set(rec1)) > 1:
        from scipy.stats import spearmanr
        spearman = float(spearmanr(sep, rec1).statistic)
    twin_ok = None
    if all(_identity(view, t) for t in twins) and len(twins) == 2:
        a, b = twins
        ia, ib = _identity(view, a), _identity(view, b)
        twin_ok = (_f(ia.get("eer_hard"), 0) > TARGETS["twin_eer"]
                   and _f(ib.get("eer_hard"), 0) > TARGETS["twin_eer"]
                   and _mentions(ia.get("confusable_with"), b)
                   and _mentions(ib.get("confusable_with"), a))
    t9, t9b, l6, t19 = [], [], [], []
    for row, exp in zip(view.truth, view.row_exp):
        base = base_id(row.get("scenario_id"))
        t0 = _f(row.get("t_start"))
        t1 = _f(row.get("t_end"), t0) + grace_s(view.dt)
        evs = [e for k in exp for e in view.ev_by_key.get(k, [])
               if t0 - view.dt <= float(e["ts"]) <= t1]
        s = str(row.get("system") or "")
        if base == "T9":
            want = _key(s, row.get("looks_like") or DEFAULT_LOOKS_LIKE["T9"])
            t9.append(any(e.get("kind") == "identity_mismatch" and _mentions(
                {k: v for k, v in (e.get("extra") or {}).items()
                 if k in ("looks_like", "best_other", "looks_like_entity", "candidate")}, want)
                for e in evs))
        elif base == "T9b":
            t9b.append(any(e.get("kind") == "unknown_identity" for e in evs))
        elif base == "L6":
            pairs = row.get("link_pairs") or DEFAULT_LINK_PAIRS["L6"]
            negs = [_key(s, n) for n in (row.get("negative_control") or DEFAULT_NEGATIVE["L6"])]
            found = 0
            for a, b in pairs:
                ka, kb = _key(s, a), _key(s, b)
                linked = kb in view.aliases.get(ka, set()) or any(
                    e.get("kind") == "entity_resolution"
                    and _f((e.get("extra") or {}).get("conf", (e.get("extra") or {}).get(
                        "confidence", 1.0)), 1.0) >= 0.9
                    and _mentions(e.get("extra") or {}, ka if _key(e["system"], e["entity"]) == kb
                                  else kb)
                    for k in (ka, kb) for e in view.ev_by_key.get(k, []))
                found += int(linked)
            neg_linked = any(view.aliases.get(n) for n in negs) or any(
                e.get("kind") == "entity_resolution" and (
                    _key(e["system"], e["entity"]) in negs
                    or any(_mentions(e.get("extra") or {}, n) for n in negs))
                for evs2 in view.ev_by_key.values() for e in evs2)
            l6.append({"recall": found / max(1, len(pairs)), "negative_linked": bool(neg_linked)})
        elif base == "T19":
            keys = set(row_keys(row))
            chain = any(keys <= act for act in view.actors) or any(
                keys <= view.inc_keys(i) for i in view.incidents)
            t19.append(bool(chain))
    return {
        "n_individuated": len(indiv),
        "window_top1": _median(win), "window_top1_mean": float(np.mean(win)) if win else None,
        "tick_top1": float(np.mean(tick)) if tick else None,
        "eer_frac_ok": (float(np.mean([e <= TARGETS["id_eer"] for e in eer])) if eer else None),
        "spearman": spearman, "twins_ok": twin_ok,
        "t9": t9, "t9b": t9b, "l6": l6, "t19": t19,
    }


def _labels(assign: Mapping[str, Any], keys: Sequence[str], fld: str) -> List[str]:
    out = []
    for k in keys:
        v = (assign.get(k) or {}).get(fld)
        s = "" if v is None else str(v)
        out.append(f"__noise__{k}" if s.lower() in NOISE_ROLES else s)
    return out


def class_stats(view: RunView) -> Dict[str, Any]:
    from sklearn.metrics import adjusted_rand_score
    assign = (view.models.get("class") or {}).get("assign") or {}
    keys = sorted(k for k in assign if k in view.personas)
    out: Dict[str, Any] = {"n": len(keys)}
    if len(keys) >= 2:
        true = [str(view.personas[k].get("archetype", "")) for k in keys]
        pred = _labels(assign, keys, "role")
        out["ari"] = float(adjusted_rand_score(true, pred))
        sizes: Dict[str, int] = {}
        for p in pred:
            sizes[p] = sizes.get(p, 0) + 1
        out["noise_frac"] = float(np.mean([p.startswith("__noise__") or sizes[p] == 1
                                           for p in pred]))
        indiv = [str(((view.personas[k].get("params") or {}).get("persona_id")) or
                     ("twin" if k in _twins(view) else k)) for k in keys]
        sub = [f"{r}/{s}" for r, s in zip(pred, _labels(assign, keys, "sub"))]
        clusters: Dict[str, Dict[str, int]] = {}
        for c, t in zip(sub, indiv):
            clusters.setdefault(c, {}).setdefault(t, 0)
            clusters[c][t] += 1
        out["sub_purity"] = float(sum(max(v.values()) for v in clusters.values()) / len(keys))
    hist = view.models.get("class_history") or []
    aris, churn = [], 0
    for a, b in zip(hist, hist[1:]):
        common = sorted(set(a.get("assign") or {}) & set(b.get("assign") or {}))
        if len(common) < 2:
            continue
        la, lb = _labels(a["assign"], common, "role"), _labels(b["assign"], common, "role")
        ari = float(adjusted_rand_score(la, lb))
        aris.append(ari)
        if ari >= 1.0 - 1e-12 and la != lb:
            churn += 1
    out["refit_ari_min"] = min(aris) if aris else None
    out["refit_ari_median"] = _median(aris)
    out["id_churn"] = churn if aris else None
    l8 = []
    for row in view.truth:
        if base_id(row.get("scenario_id")) != "L8":
            continue
        t0 = _f(row.get("t_start"))
        for k in row_keys(row):
            t_first = view.first_seen.get(k, t0)
            ok = any(e.get("kind") == "new_entity_matched"
                     and _f((e.get("extra") or {}).get("prob", (e.get("extra") or {}).get(
                         "probability", e.get("score"))), 0.0) >= TARGETS["l8_prob"]
                     and float(e["ts"]) <= t_first + 2 * view.dt + 1e-6
                     for e in view.ev_by_key.get(k, []))
            l8.append(ok)
    out["l8"] = l8
    return out


# --------------------------------------------------------------------------- #
# Explanation (gate 10) and portrait (gate 11)
# --------------------------------------------------------------------------- #
RANGE_KEYS = ("range", "usual", "usual_range", "interval", "natural_range", "normal_range",
              "p5", "band")
_RANGE_RE = re.compile(r"\d[\d.,]*\s*(?:[a-zA-Z/%]+\s*)?[–~-]\s*\d")


def top_features(expl: Mapping[str, Any], n: int = 3) -> List[str]:
    attr = expl.get("attributions")
    if attr is None:
        attr = expl.get("top_features") or []
    if isinstance(attr, Mapping):
        items = sorted(attr.items(), key=lambda kv: -_f(kv[1] if not isinstance(kv[1], Mapping)
                                                         else kv[1].get("share"), 0.0))
        return [_norm_feature(k) for k, _ in items[:n]]
    out = []
    for a in attr:
        if isinstance(a, (list, tuple)) and a:
            out.append(_norm_feature(a[0]))
        else:
            out.append(_norm_feature(a))
    return out[:n]


def has_natural_range(inc: Mapping[str, Any]) -> bool:
    expl = inc.get("explanation") or {}
    if any(k in expl for k in ("natural_range", "normal_range")):
        return True
    attr = expl.get("attributions")
    items = attr.values() if isinstance(attr, Mapping) else (attr or [])
    for a in items:
        if isinstance(a, Mapping) and any(k in a for k in RANGE_KEYS):
            return True
    texts = [inc.get("narrative") or "", expl.get("narrative_en") or "",
             expl.get("narrative_zh") or ""]
    return any(_RANGE_RE.search(str(t)) for t in texts)


def explanation_stats(view: RunView, outcomes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_id = {str(i.get("id")): i for i in view.incidents}
    hits, cf = [], []
    for o in outcomes:
        row = next((r for r in view.truth if str(r.get("scenario_id")) == o["scenario_id"]), {})
        pf = {_norm_feature(f) for f in row.get("perturbed_features") or []}
        for iid in o.get("tp_incidents") or []:
            inc = by_id.get(iid)
            if inc is None:
                continue
            expl = inc.get("explanation") or {}
            if pf:
                hits.append(bool(set(top_features(expl, 3)) & pf))
            v = expl.get("counterfactual_valid")
            if v is not None:
                cf.append(bool(v))
    scen = [i for i in view.incidents if view.t0 <= float(i.get("opened", 0.0)) <= view.t1]
    nat = [has_natural_range(i) for i in scen]
    return {"hit3": hits, "cf_valid": cf, "natural_range": nat}


def _hours_from(v: Any) -> Optional[Set[int]]:
    if v is None:
        return None
    if isinstance(v, str):
        m = re.findall(r"(\d{1,2}):(\d{2})", v)
        if len(m) >= 2:
            a = int(m[0][0]) + int(m[0][1]) / 60.0
            b = int(m[1][0]) + int(m[1][1]) / 60.0
            return _span_hours(a, b)
        return None
    if isinstance(v, Mapping):
        for k in ("hours", "typical_hours", "active_hours", "window"):
            if k in v:
                return _hours_from(v[k])
        if "start" in v and "end" in v:
            return _span_hours(_f(v["start"]), _f(v["end"]))
        return None
    if isinstance(v, (list, tuple)):
        if len(v) == 2 and all(isinstance(x, (int, float)) for x in v) and \
                not all(float(x).is_integer() and x < 24 for x in v):
            return _span_hours(float(v[0]), float(v[1]))
        if all(isinstance(x, (int, float)) for x in v):
            return {int(x) % 24 for x in v}
    return None


def _span_hours(a: float, b: float) -> Set[int]:
    if math.isnan(a) or math.isnan(b):
        return set()
    return {h % 24 for h in range(int(math.floor(a)), int(math.ceil(b)))} if b > a else \
        {h % 24 for h in range(int(math.floor(a)), int(math.ceil(b + 24)))}


def _portrait_json(por: Any) -> Mapping[str, Any]:
    if isinstance(por, Mapping):
        j = por.get("json", por)
        return j if isinstance(j, Mapping) else {}
    return {}


def _portrait_hours(j: Mapping[str, Any]) -> Optional[Set[int]]:
    for k in ("typical_hours", "active_hours", "hours", "window"):
        if k in j:
            return _hours_from(j[k])
    r = j.get("rhythm")
    return _hours_from(r) if r is not None else None


def _persona_hours(meta: Mapping[str, Any]) -> Optional[Set[int]]:
    p = meta.get("params") or {}
    for a, b in (("work_start", "work_end"), ("start", "end"), ("window_start", "window_end")):
        if a in p and b in p:
            return _span_hours(_f(p[a]), _f(p[b]))
    if "window" in p:
        return _hours_from(p["window"])
    return None


def _names_of(v: Any) -> List[str]:
    if isinstance(v, Mapping):
        return [str(k) for k, _ in sorted(v.items(), key=lambda kv: -_f(kv[1], 0.0))]
    out = []
    for x in v or []:
        if isinstance(x, Mapping):
            out.append(str(x.get("template") or x.get("name") or x.get("token") or ""))
        elif isinstance(x, (list, tuple)) and x:
            out.append(str(x[0]))
        else:
            out.append(str(x))
    return out


def portrait_stats(view: RunView) -> Dict[str, Any]:
    jac, trec = [], []
    for k, meta in view.personas.items():
        if k not in view.control:
            continue
        j = _portrait_json(((view.profiles.get(k) or {}).get("extra") or {}).get("portrait"))
        if not j:
            continue
        if str(meta.get("archetype", "")).lower() in HUMAN_ARCHETYPES:
            ph, th = _portrait_hours(j), _persona_hours(meta)
            if ph is not None and th:
                jac.append(len(ph & th) / max(1, len(ph | th)))
        p = meta.get("params") or {}
        true_t = _names_of(p.get("path_pref") or p.get("top_templates") or [])[:5]
        got = _names_of(j.get("top_templates") or [])[:5]
        if true_t and got:
            trec.append(len(set(true_t) & set(got)) / len(true_t))
    cov = []
    ho = _get(view.run, "holdout", None) or {}
    names = list(ho.get("feature_names") or [])
    for k, rows in (ho.get("nat") or {}).items():
        if k not in view.control:
            continue
        j = _portrait_json((ho.get("portraits") or {}).get(k))
        bands = j.get("workload") or j.get("bands") or {}
        if not isinstance(bands, Mapping) or not names:
            continue
        V = np.asarray(rows.get("values"), dtype=float)
        for f, b in bands.items():
            fn = _norm_feature(f)
            if fn not in names:
                continue
            lo = _f(b.get("p5")) if isinstance(b, Mapping) else _f(b[0]) if b else math.nan
            hi = _f(b.get("p95")) if isinstance(b, Mapping) else _f(b[-1]) if b else math.nan
            if math.isnan(lo) or math.isnan(hi):
                continue
            x = V[:, names.index(fn)]
            x = x[np.isfinite(x)]
            cov.extend(((x >= lo) & (x <= hi)).tolist())
    class_keys = set()
    assign = (view.models.get("class") or {}).get("assign") or {}
    role_count: Dict[str, int] = {}
    for k, a in assign.items():
        for ck in RunView._class_keys_of(k, a):
            role_count[ck] = role_count.get(ck, 0) + 1
    class_keys = {ck for ck, n in role_count.items() if n >= 2 or ":static:" in ck
                  or ":pool:" in ck}
    have = [bool(((view.profiles.get(ck) or {}).get("extra") or {}).get("portrait"))
            for ck in sorted(class_keys)]
    return {"jaccard": jac, "template_recall": trec,
            "coverage": float(np.mean(cov)) if cov else None, "n_coverage": len(cov),
            "class_portraits": have}


# --------------------------------------------------------------------------- #
# Robustness (gate 15) and CPU / memory (gate 14)
# --------------------------------------------------------------------------- #
def robustness_stats(view: RunView) -> Dict[str, Any]:
    run = view.run
    return {
        "exceptions": len(_get(run, "exceptions", None) or []),
        "exception_heads": [e.get("engine", "") + ": " + e.get("error", "")
                            for e in (_get(run, "exceptions", None) or [])[:10]],
        "stale": len(_get(run, "stale", None) or []),
        "stale_heads": [f"{s.get('system')}|{s.get('entity')}|{s.get('name')}"
                        for s in (_get(run, "stale", None) or [])[:10]],
        "pipeline_degraded": sum(1 for e in view.events if e.get("kind") == "pipeline_degraded"),
        "aborted": _get(run, "aborted", None),
        "ablated": bool(_get(run, "disabled_engines", None)),
    }


def cpu_mem_stats(view: RunView) -> Dict[str, Any]:
    run = view.run
    t = _get(run, "timings", None) or {}
    ticks = np.asarray(t.get("ticks", np.zeros((0, 11))), dtype=float)
    cols = list(t.get("columns") or [])
    out: Dict[str, Any] = {"wall_s": float(_get(run, "wall_s", 0.0) or 0.0)}
    if ticks.size and cols:
        c = {n: i for i, n in enumerate(cols)}
        live = ticks[:, c["training"]] == 0
        beh = ticks[:, c["behavior_ms"]]
        n_ent = max(1, len(view.real))
        out.update({
            "n_ticks": int(ticks.shape[0]), "n_entities": n_ent,
            "pipeline_s": float(ticks[:, c["pipeline_ms"]].sum() / 1000.0),
            "lib3_s": float(beh.sum() / 1000.0),
            "gen_s": float(ticks[:, c["gen_ms"]].sum() / 1000.0),
            "collect_s": float(ticks[:, c["collect_ms"]].sum() / 1000.0),
            "lib3_p95_ms": float(np.quantile(beh[live], 0.95)) if live.any() else None,
            "lib3_p95_ms_at35": (float(np.quantile(beh[live], 0.95)) * 35.0 / n_ent
                                 if live.any() else None),
            "layers_ms_mean": {L: float(ticks[:, c[f"{L}_ms"]].mean())
                               for L in ("raw", "derived", "behavior", "signature")},
        })
        em = np.asarray(t.get("engine_ms", np.zeros((0, 0))), dtype=float)
        if em.size:
            names = list(t.get("engine_names") or [])
            out["engine_ms_mean"] = {n: float(em[:, i].mean()) for i, n in enumerate(names)}
            out["engine_ms_p95"] = {n: float(np.quantile(em[:, i], 0.95))
                                    for i, n in enumerate(names)}
    mem = _get(run, "memory", None) or {}
    fin = (mem.get("final") or {})
    out["bytes_per_entity"] = fin.get("bytes_per_entity")
    phases = sorted(k for k in mem if k.startswith("phase"))
    if len(phases) >= 2 and view.t1 > view.t0:
        warm = mem[phases[-2]].get("bytes_per_entity")
        live_days = (view.t1 - view.t0) / 86400.0
        if warm is not None and fin.get("bytes_per_entity") is not None and live_days >= 0.5:
            grow = (fin["bytes_per_entity"] - warm) / live_days
            out["bytes_per_entity_8d"] = float(warm + max(0.0, grow) * 8.0)
    return out


# --------------------------------------------------------------------------- #
# score_run: everything per run, JSON-able
# --------------------------------------------------------------------------- #
def score_run(run: Any) -> Dict[str, Any]:
    """All per-run measurements of one RunResult as a JSON-able dict."""
    view = RunView(run)
    scen, legit = [], []
    for i, row in enumerate(view.truth):
        if row_label(row) == "malicious":
            scen.append(scenario_outcome(view, i))
        elif row_label(row) in ("legit_change", "system_change"):
            legit.append(legit_outcome(view, i))
    return {
        "pack": view.pack, "seed": view.seed, "dt": view.dt,
        "scenario_window": [view.t0, view.t1],
        "disabled_engines": list(_get(run, "disabled_engines", None) or []),
        "labels_added": int(_get(run, "labels_added", 0) or 0),
        "aborted": _get(run, "aborted", None),
        "scenarios": scen,
        "legit": legit,
        "far": far_counts(view),
        "range": range_scores(view),
        "calibration": calibration_stats(view),
        "poisoning": poisoning_checks(view, scen),
        "identification": identification_stats(view),
        "classes": class_stats(view),
        "explanation": explanation_stats(view, scen),
        "portrait": portrait_stats(view),
        "robustness": robustness_stats(view),
        "cpu_mem": cpu_mem_stats(view),
    }


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
def gate(name: str, value: Any, target: Any, checks: Sequence[Dict[str, Any]],
         **details: Any) -> Dict[str, Any]:
    """A gate record. pass = every computable check passes; None (n/a) when
    the headline check (the first) cannot be computed, so a gate never
    'passes' on side checks alone (e.g. calibration with no behavior.p)."""
    vals = [c["pass"] for c in checks if c.get("pass") is not None]
    if not vals or (checks and checks[0].get("pass") is None and all(vals)):
        verdict: Optional[bool] = None
    else:
        verdict = all(vals)
    return {"name": name, "value": value, "target": target, "pass": verdict,
            "details": {"checks": list(checks), **details}}


def check(name: str, value: Any, target: Any, passed: Optional[bool],
          ci: Optional[Tuple[Optional[float], Optional[float]]] = None) -> Dict[str, Any]:
    d = {"name": name, "value": value, "target": target,
         "pass": None if passed is None else bool(passed)}
    if ci is not None:
        d["ci95"] = list(ci)
    return d


def _le(v: Optional[float], t: float) -> Optional[bool]:
    return None if v is None else v <= t + 1e-12


def _ge(v: Optional[float], t: float) -> Optional[bool]:
    return None if v is None else v >= t - 1e-12


def _frac(xs: Iterable[Any]) -> Optional[float]:
    x = [bool(v) for v in xs if v is not None]
    return float(np.mean(x)) if x else None


def _by_seed(scores: Sequence[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for s in scores:
        out.setdefault(int(s.get("seed", 0)), []).append(s)
    return out


def _recall(outcomes: Sequence[Dict[str, Any]], within: bool = True) -> Optional[float]:
    if not outcomes:
        return None
    return float(np.mean([o["within_deadline"] if within else o["detected"] for o in outcomes]))


def _seed_values(scores: Sequence[Dict[str, Any]],
                 fn: Callable[[List[Dict[str, Any]]], Optional[float]]) -> List[float]:
    return [v for v in (fn(g) for g in _by_seed(scores).values()) if v is not None]


def gate_detection(scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    outs = [o for s in scores for o in s["scenarios"]]
    loud = [o for o in outs if o["loudness"] == "loud"]
    subtle = [o for o in outs if o["loudness"] == "subtle"]
    threat = [o for o in outs if o["loudness"] != "p2"]
    r_loud, r_sub, r_all = _recall(loud), _recall(subtle), _recall(threat)

    def seed_recall(sc: List[Dict[str, Any]]) -> Optional[float]:
        return _recall([o for s in sc for o in s["scenarios"] if o["loudness"] != "p2"])

    per_seed = _seed_values(scores, seed_recall)
    rng = [s["range"]["range"] for s in scores if s.get("range")]
    pa = [s["range"]["point_adjusted"] for s in scores if s.get("range")]
    checks = [
        check("loud recall (all seeds)", r_loud, TARGETS["loud_recall"],
              _ge(r_loud, TARGETS["loud_recall"])),
        check("subtle recall (pooled, within deadline)", r_sub, TARGETS["subtle_recall"],
              _ge(r_sub, TARGETS["subtle_recall"])),
        check("overall threat recall", r_all, TARGETS["overall_recall"],
              _ge(r_all, TARGETS["overall_recall"]), bootstrap_ci(per_seed)),
    ]
    rp = {k: _median(r.get(k) for r in rng) for k in ("precision", "recall", "f1")}
    pp = {k: _median(r.get(k) for r in pa) for k in ("precision", "recall", "f1")}
    checks.append(check("range F1 (Tatbul 2018; report)", rp["f1"], None, None,
                        bootstrap_ci([r.get("f1") for r in rng])))
    checks.append(check("point-adjusted F1 (reference only)", pp["f1"], None, None))
    return gate("Detection", r_all, TARGETS["overall_recall"], checks,
                recall_raw=_recall(threat, within=False), range=rp, point_adjusted=pp,
                n_loud=len(loud), n_subtle=len(subtle),
                missed=[f"{o['pack']}/{o['seed']}/{o['scenario_id']}" for o in threat
                        if not o["within_deadline"]][:50])


def ttd_table(scores: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for s in scores:
        for o in s["scenarios"]:
            groups.setdefault((s["pack"], o["scenario_id"]), []).append(o)
    rows = []
    for (pack, sid), os_ in sorted(groups.items()):
        det = [o for o in os_ if o["detected"]]
        rows.append({
            "pack": pack, "scenario_id": sid, "base": base_id(sid), "dt": os_[0]["dt"],
            "loudness": os_[0]["loudness"], "n_runs": len(os_), "n_detected": len(det),
            "recall": len(det) / len(os_),
            "within_deadline": float(np.mean([o["within_deadline"] for o in os_])),
            "ttd_ticks_median": _median(o["ttd_ticks"] for o in det),
            "ttd_ticks_p90": _quantile((o["ttd_ticks"] for o in det), 0.9),
            "ttd_s_median": _median(o["ttd_s"] for o in det),
            "ttd_s_p90": _quantile((o["ttd_s"] for o in det), 0.9),
            "ttd_s_ci95": list(bootstrap_ci([o["ttd_s"] for o in det])),
            "max_severity": max((o["max_severity"] for o in os_ if o["max_severity"]),
                                key=sev_rank, default=None),
            "required_severity": os_[0]["required_severity"],
            "deadline_s": os_[0]["deadline_s"],
        })
    return rows


def gate_ttd(scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    table = ttd_table(scores)
    checks = []
    for r in table:
        if r["loudness"] != "loud":
            continue
        if abs(r["dt"] - 900.0) < 1:
            v, t = r["ttd_ticks_median"], TARGETS["loud_ttd_ticks"]
        else:
            v, t = r["ttd_s_median"], TARGETS["loud_ttd_wall_s"]
        checks.append(check(f"loud TTD {r['pack']}/{r['scenario_id']}", v, t,
                            False if v is None else v <= t))
    by_base: Dict[Tuple[str, str], Optional[float]] = {}
    for r in table:
        by_base[(r["pack"], r["base"])] = r["ttd_s_median"]
    pack_a = next((p for p, _ in by_base if pack_is(p, "A")), None)
    pack_e = next((p for p, _ in by_base if pack_is(p, "E")), None)
    if pack_a and pack_e:
        for b in PACK_E_REPLAYS:
            a, e = by_base.get((pack_a, b)), by_base.get((pack_e, b))
            if a is None or e is None:
                continue
            checks.append(check(f"Pack E {b} wall TTD <= Pack A + 15 min", e,
                                a + TARGETS["pack_e_slack_s"],
                                e <= a + TARGETS["pack_e_slack_s"] + 1e-6))
    all_ticks = [r["ttd_ticks_median"] for r in table if r["ttd_ticks_median"] is not None]
    return gate("Time to detect", _median(all_ticks), "loud <= 2 ticks @900 s; E <= A + 15 min",
                checks, table=table)


def _pooled_far(scores: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    days = sum(s["far"]["entity_days"] for s in scores)
    return sum(s["far"][key] for s in scores) / days if days > 0 else None


def gate_far(scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    low, med = _pooled_far(scores, "n_low"), _pooled_far(scores, "n_medium")
    n_high = sum(s["far"]["n_high"] for s in scores)
    n_crit = sum(s["far"]["n_critical"] for s in scores)
    seed_low = _seed_values(scores, lambda g: _pooled_far(g, "n_low"))
    seed_med = _seed_values(scores, lambda g: _pooled_far(g, "n_medium"))
    checks = [
        check("FAR >= LOW per entity-day", low, TARGETS["far_low"],
              _le(low, TARGETS["far_low"]), bootstrap_ci(seed_low)),
        check("FAR >= MEDIUM per entity-day", med, TARGETS["far_medium"],
              _le(med, TARGETS["far_medium"]), bootstrap_ci(seed_med)),
        check("HIGH+ total across runs", n_high, TARGETS["far_high_total"],
              n_high <= TARGETS["far_high_total"] if scores else None),
        check("CRITICAL total", n_crit, TARGETS["far_critical"],
              n_crit <= TARGETS["far_critical"] if scores else None),
    ]
    g = {"bursty": [0, 0.0], "steady": [0, 0.0]}
    for s in scores:
        for name, v in s["far"]["groups"].items():
            if name in g:
                g[name][0] += v["n_low"]
                g[name][1] += v["days"]
    ratio = None
    if g["bursty"][1] > 0 and g["steady"][1] > 0 and (g["bursty"][0] + g["steady"][0]) >= 2:
        rb = (g["bursty"][0] + 0.5) / g["bursty"][1]
        rs = (g["steady"][0] + 0.5) / g["steady"][1]
        ratio = max(rb, rs) / min(rb, rs)
    checks.append(check("bursty vs steady FAR ratio", ratio, TARGETS["bursty_steady_ratio"],
                        _le(ratio, TARGETS["bursty_steady_ratio"])))
    pa = [s for s in scores if pack_is(s["pack"], "A")]
    pe = [s for s in scores if pack_is(s["pack"], "E")]
    cad = None
    if pa and pe:
        fa, fe = _pooled_far(pa, "n_low"), _pooled_far(pe, "n_low")
        if fa and fe is not None:
            cad = fe / fa
        elif fa == 0 and fe == 0:
            cad = 1.0
    lo, hi = TARGETS["cadence_ratio"]
    checks.append(check("cadence invariance FAR(E@60s)/FAR(A@900s)", cad, [lo, hi],
                        None if cad is None else lo <= cad <= hi))
    table = [{"pack": s["pack"], "seed": s["seed"], "entity_days": s["far"]["entity_days"],
              "n_low": s["far"]["n_low"], "n_medium": s["far"]["n_medium"],
              "n_high": s["far"]["n_high"], "n_critical": s["far"]["n_critical"],
              "far_low": s["far"]["far_low"], "far_medium": s["far"]["far_medium"]}
             for s in scores]
    return gate("False alarms on control entities", low, TARGETS["far_low"], checks,
                table=table, groups=g)


def gate_legit(scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [r for s in scores for r in s["legit"]]
    frac = _frac(r["sev_ok"] for r in rows)
    risk = _frac(r["risk_ok"] for r in rows)
    checks = [check("seed x scenario runs within allowed severity", frac,
                    TARGETS["legit_ok_frac"], _ge(frac, TARGETS["legit_ok_frac"]),
                    bootstrap_ci(_seed_values(scores, lambda g: _frac(
                        r["sev_ok"] for s in g for r in s["legit"])))),
              check("affected entities / classes max risk < 30 (fraction)", risk, 1.0,
                    _ge(risk, 1.0))]
    return gate("Legitimate changes", frac, TARGETS["legit_ok_frac"], checks,
                table=[{k: r[k] for k in ("pack", "seed", "scenario_id", "max_severity",
                                          "allowed", "max_risk", "ok")} for r in rows])


def gate_burden(scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    outs = [o for s in scores for o in s["scenarios"] if o["detected"]]
    notif = _median([]) if not outs else float(np.mean([o.get("notifications", 1)
                                                        for o in outs]))
    ipe = float(np.mean([o["incidents_per_episode"] for o in outs])) if outs else None
    cls = [r["n_class_incidents"] for s in scores for r in s["legit"] if r["base"] in CLASS_LEGIT]
    mx = max(cls) if cls else None
    checks = [check("notifications per TP incident", notif, TARGETS["notif_per_tp"],
                    _le(notif, TARGETS["notif_per_tp"])),
              check("incidents per episode", ipe, TARGETS["incidents_per_episode"],
                    _le(ipe, TARGETS["incidents_per_episode"])),
              check("class incidents per class-wide legit event (max)", mx,
                    TARGETS["class_incidents_per_event"],
                    _le(mx, TARGETS["class_incidents_per_event"]))]
    return gate("Alert burden", notif, TARGETS["notif_per_tp"], checks)


def gate_poisoning(scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    recs = [r for s in scores for r in s["poisoning"]["scenarios"]]
    acc = [a for s in scores for a in s["poisoning"]["accepts"]]

    def frac(key: str) -> Optional[float]:
        return _frac(r.get(key) for r in recs)

    bumps = sum(r["control_bumps"] for r in recs)
    ks = [r["null_ks"] for r in recs if r.get("null_ks") is not None]
    drift = sorted({k for s in scores for k in s["poisoning"]["long_drift"]})
    checks = [
        check("current anchor within 0.3 sigma15 of pre-onset (fraction)", frac("current_ok"),
              1.0, _ge(frac("current_ok"), 1.0)),
        check("reference anchor within 0.1 sigma15 (fraction)", frac("reference_ok"), 1.0,
              _ge(frac("reference_ok"), 1.0)),
        check("golden unchanged (fraction)", frac("golden_ok"), 1.0, _ge(frac("golden_ok"), 1.0)),
        check("model.control bumps inside malicious windows", bumps, 0,
              bumps == 0 if recs else None),
        check("incident closed within max(8 ticks, 2 h) of attack end", frac("closed_ok"),
              1.0, _ge(frac("closed_ok"), 1.0)),
        check("null KS D after attack (median)", _median(ks), TARGETS["ks_d"],
              _le(_median(ks), TARGETS["ks_d"])),
        check("legit accepts within deadline (fraction)", _frac(a["ok"] for a in acc), 1.0,
              _ge(_frac(a["ok"] for a in acc), 1.0)),
        check("entities DRIFTING > 14 d without label queue", len(drift), 0,
              len(drift) == 0 if scores else None),
        check("rollback replay == offline fit within 1e-6", None, 1e-6, None),
    ]
    return gate("Poisoning and reversibility", bumps, 0, checks, long_drift=drift)


def gate_calibration(scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    per_det: Dict[str, List[float]] = {}
    for s in scores:
        for d, v in s["calibration"]["ks"].items():
            if v.get("D") is not None and v.get("n", 0) >= 100:
                per_det.setdefault(d, []).append(v["D"])
    ks_med = {d: _median(v) for d, v in per_det.items()}
    worst = max(ks_med.values()) if ks_med else None
    checks = [check("max over detectors of median KS D", worst, TARGETS["ks_d"],
                    _le(worst, TARGETS["ks_d"]))]
    lo, hi = TARGETS["exceed_ratio"]
    by_cc: Dict[int, Dict[str, List[float]]] = {}
    for s in scores:
        c = s["calibration"]
        for x, v in c["exceedance"].items():
            agg = by_cc.setdefault(int(c["cc"]), {}).setdefault(x, [0, 0.0])
            agg[0] += v["k"]
            agg[1] += v["expected"]
    for cc, xs in sorted(by_cc.items()):
        for x, (k, e) in sorted(xs.items()):
            ratio = k / e if e > 0 else None
            checks.append(check(f"exceedance e_day<={x} cc={cc} (observed/nominal)", ratio,
                                [lo, hi], None if (ratio is None or e < 5) else lo <= ratio <= hi))
    days = sum(s["calibration"]["entity_days"] for s in scores)
    ev = sum(s["calibration"]["evidence_cusum_alarms"] for s in scores)
    ev_rate = ev / days if days > 0 else None
    checks.append(check("evidence-CUSUM alarms per entity-day", ev_rate,
                        TARGETS["evidence_cusum_rate"], _le(ev_rate, TARGETS["evidence_cusum_rate"])))
    paths: Dict[str, int] = {}
    for s in scores:
        for p, n in s["calibration"]["acc_path_alarms"].items():
            paths[p] = paths.get(p, 0) + n
    for p, n in sorted(paths.items()):
        budget = BUDGET_PATHS.get(p)
        if budget is None or days <= 0 or p == "class":
            continue
        rate = n / days
        checks.append(check(f"accumulator path '{p}' rate per entity-day", rate,
                            TARGETS["family_rate_mult"] * budget,
                            rate <= TARGETS["family_rate_mult"] * budget + 1e-12))
    acat = sum(s["calibration"]["acat_incident_ticks"] for s in scores)
    checks.append(check("ACAT masking at incident ticks", acat, 0, acat == 0 if scores else None))
    return gate("Calibration", worst, TARGETS["ks_d"], checks, ks_by_detector=ks_med,
                acat_ticks_all=sum(s["calibration"]["acat_ticks"] for s in scores))


def gate_identification(scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ids = [s["identification"] for s in scores]
    win = _median(i["window_top1_mean"] for i in ids)
    tick = _median(i["tick_top1"] for i in ids)
    eer = _median(i["eer_frac_ok"] for i in ids)
    sp = _median(i["spearman"] for i in ids)
    tw = _frac(i["twins_ok"] for i in ids)
    t9 = _frac(v for i in ids for v in i["t9"])
    t9b = _frac(v for i in ids for v in i["t9b"])
    l6 = [x for i in ids for x in i["l6"]]
    l6_rec = _median(x["recall"] for x in l6)
    l6_neg = any(x["negative_linked"] for x in l6) if l6 else None
    t19 = _frac(v for i in ids for v in i["t19"])
    checks = [
        check("window top-1 (K=4)", win, TARGETS["id_window_top1"],
              _ge(win, TARGETS["id_window_top1"])),
        check("per-tick top-1 @900 s", tick, TARGETS["id_tick_top1"],
              _ge(tick, TARGETS["id_tick_top1"])),
        check("personas with EER_hard <= 0.05", eer, TARGETS["id_eer_frac"],
              _ge(eer, TARGETS["id_eer_frac"])),
        check("twins confusable (EER_hard > 0.2, listed)", tw, 1.0, _ge(tw, 1.0)),
        check("Spearman(separability, CV recall)", sp, TARGETS["sep_spearman"],
              _ge(sp, TARGETS["sep_spearman"])),
        check("T9 looks_like correct", t9, 1.0, _ge(t9, 1.0)),
        check("T9b unknown_identity", t9b, TARGETS["t9b_frac"], _ge(t9b, TARGETS["t9b_frac"])),
        check("L6 link recall", l6_rec, TARGETS["l6_recall"], _ge(l6_rec, TARGETS["l6_recall"])),
        check("L6 negative control never linked", None if l6_neg is None else not l6_neg, True,
              None if l6_neg is None else not l6_neg),
        check("T19 actor chain recovered", t19, TARGETS["t19_frac"], _ge(t19, TARGETS["t19_frac"])),
    ]
    return gate("Identification", win, TARGETS["id_window_top1"], checks)


def gate_classes(scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    cs = [s["classes"] for s in scores]
    ari = _median(c.get("ari") for c in cs)
    noise = _median(c.get("noise_frac") for c in cs)
    pur = _median(c.get("sub_purity") for c in cs)
    rari = _median(c.get("refit_ari_min") for c in cs)
    churn = [c.get("id_churn") for c in cs if c.get("id_churn") is not None]
    l8 = _frac(v for c in cs for v in c.get("l8", []))
    t21 = _frac(o["within_deadline"] for s in scores for o in s["scenarios"] if o["base"] == "T21")
    cl = [r for s in scores for r in s["legit"] if r["base"] in ("L1", "L2", "L3")]
    cl_ok = _frac(sev_rank(r["class_max_severity"]) <= SEV_RANK["low"] if r["class_max_severity"]
                  else True for r in cl)
    checks = [
        check("level-1 role ARI vs archetypes", ari, TARGETS["ari"], _ge(ari, TARGETS["ari"]),
              bootstrap_ci([c.get("ari") for c in cs])),
        check("noise/unique fraction", noise, TARGETS["noise_frac"],
              _le(noise, TARGETS["noise_frac"])),
        check("level-2 sub-class purity", pur, TARGETS["sub_purity"],
              _ge(pur, TARGETS["sub_purity"])),
        check("ARI between consecutive refits (min)", rari, TARGETS["refit_ari"],
              _ge(rari, TARGETS["refit_ari"])),
        check("ID churn with unchanged membership", sum(churn) if churn else None, 0,
              (sum(churn) == 0) if churn else None),
        check("L8 typing prob >= 0.8 within 3 ticks", l8, 1.0, _ge(l8, 1.0)),
        check("T21 class detection", t21, TARGETS["t21_frac"], _ge(t21, TARGETS["t21_frac"])),
        check("L1-L3 class incidents <= LOW", cl_ok, 1.0, _ge(cl_ok, 1.0)),
    ]
    return gate("Classes", ari, TARGETS["ari"], checks)


def gate_explanation(scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ex = [s["explanation"] for s in scores]
    hit = _frac(v for e in ex for v in e["hit3"])
    cf = _frac(v for e in ex for v in e["cf_valid"])
    nat = _frac(v for e in ex for v in e["natural_range"])
    checks = [check("hit@3 vs perturbed_features", hit, TARGETS["hit3"], _ge(hit, TARGETS["hit3"])),
              check("counterfactual validity", cf, TARGETS["cf_valid"], _ge(cf, TARGETS["cf_valid"])),
              check("natural-unit range present", nat, TARGETS["nat_range"],
                    _ge(nat, TARGETS["nat_range"]))]
    return gate("Explanation", hit, TARGETS["hit3"], checks)


def gate_portrait(scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ps = [s["portrait"] for s in scores]
    jac = _median(v for p in ps for v in p["jaccard"])
    tr = _median(v for p in ps for v in p["template_recall"])
    cov = _median(p["coverage"] for p in ps)
    cp = _frac(v for p in ps for v in p["class_portraits"])
    lo, hi = TARGETS["coverage"]
    checks = [check("typical-hours Jaccard", jac, TARGETS["portrait_jaccard"],
                    _ge(jac, TARGETS["portrait_jaccard"])),
              check("top-5 template recall", tr, TARGETS["template_recall"],
                    _ge(tr, TARGETS["template_recall"])),
              check("p5-p95 coverage of held-out ticks", cov, [lo, hi],
                    None if cov is None else lo <= cov <= hi),
              check("class portraits exist", cp, TARGETS["class_portraits"],
                    _ge(cp, TARGETS["class_portraits"]))]
    return gate("Portrait", jac, TARGETS["portrait_jaccard"], checks)


def feedback_gate(base: Sequence[Dict[str, Any]], fb: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Gate 12 from paired runs (same pack/seed) without and with the
    simulated analyst: control incidents >= LOW must fall by >= 50% once 20
    labels were given, with recall dropping by <= 0.02."""
    if not base or not fb:
        return gate("Feedback", None, TARGETS["feedback_cut"],
                    [check("paired feedback runs", None, "run with --feedback", None)])
    n_labels = sum(s.get("labels_added", 0) for s in fb)
    b_low = sum(s["far"]["n_low"] for s in base)
    f_low = sum(s["far"]["n_low"] for s in fb)
    cut = (b_low - f_low) / b_low if b_low > 0 else None
    rb = _recall([o for s in base for o in s["scenarios"] if o["loudness"] != "p2"])
    rf = _recall([o for s in fb for o in s["scenarios"] if o["loudness"] != "p2"])
    drop = None if rb is None or rf is None else rb - rf
    enough = n_labels >= 20
    # fewer than 20 labels (too few incidents to label) leaves the gate n/a
    checks = [check("control incidents >= LOW cut", cut, TARGETS["feedback_cut"],
                    None if not enough else _ge(cut, TARGETS["feedback_cut"])),
              check("recall drop", drop, TARGETS["feedback_recall_drop"],
                    None if not enough else _le(drop, TARGETS["feedback_recall_drop"])),
              check("labels given (precondition)", n_labels, 20, None)]
    return gate("Feedback", cut, TARGETS["feedback_cut"], checks)


def ablation_table(full: Sequence[Dict[str, Any]],
                   ablated: Mapping[str, Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Per engine: Δrecall and ΔFAR (>= LOW) per scenario versus the full
    system, and the scenarios it is the sole detector of (detected with it,
    missed without it in every seed)."""
    def per_sid(sc: Sequence[Dict[str, Any]]) -> Dict[str, List[bool]]:
        out: Dict[str, List[bool]] = {}
        for s in sc:
            for o in s["scenarios"]:
                out.setdefault(f"{s['pack']}/{o['scenario_id']}", []).append(o["within_deadline"])
        return out

    f = per_sid(full)
    f_far = _pooled_far(full, "n_low")
    rows = []
    for eng, sc in sorted(ablated.items()):
        a = per_sid(sc)
        delta = {k: float(np.mean(a[k]) - np.mean(v)) for k, v in f.items() if k in a}
        sole = [k for k, v in f.items() if k in a and any(v) and not any(a[k])]
        main = [k for k, d in delta.items() if d <= -0.5]
        a_far = _pooled_far(sc, "n_low")
        rows.append({"engine": eng, "delta_recall": delta, "sole_detector_of": sole,
                     "main_detector_of": main,
                     "delta_far_low": (None if a_far is None or f_far is None else a_far - f_far)})
    return rows


def gate_ablation(full: Sequence[Dict[str, Any]],
                  ablated: Optional[Mapping[str, Sequence[Dict[str, Any]]]]) -> Dict[str, Any]:
    if not ablated:
        return gate("Ablation", None, "each P0/P1 engine sole/main detector of >= 1 scenario",
                    [check("ablation runs", None, "run with --ablate", None)])
    rows = ablation_table(full, ablated)
    have = any(s["scenarios"] for s in full)
    checks = []
    for r in rows:
        checks.append(check(f"{r['engine']} sole/main detector of >= 1 scenario",
                            len(r["sole_detector_of"]) + len(r["main_detector_of"]), 1,
                            bool(r["sole_detector_of"] or r["main_detector_of"]) if have
                            else None))
        if r["engine"].lower() in ("b18", "class_monitor"):
            t21 = any(base_id(k.partition("/")[2]) == "T21" for k in r["sole_detector_of"])
            checks.append(check("B18 is the sole detector of T21", t21, True, t21))
        if r["engine"].lower() in ("b04", "likelihood"):
            d = r["delta_far_low"]
            checks.append(check("fault injection B04 off: FAR does not rise", d, 0.0,
                                None if d is None else d <= 1e-12))
    return gate("Ablation", len(rows), "each P0/P1 engine sole/main detector of >= 1 scenario",
                checks, table=rows)


def gate_cpu_mem(scores: Sequence[Dict[str, Any]],
                 smoke: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cm = [s["cpu_mem"] for s in scores]
    walls = [c["wall_s"] for c in cm]
    worst = max(walls) if walls else None
    p95 = _median(c.get("lib3_p95_ms_at35") for c in cm)
    mem = max((c["bytes_per_entity"] for c in cm if c.get("bytes_per_entity") is not None),
              default=None)
    e8 = [c["bytes_per_entity_8d"] for s, c in zip(scores, cm)
          if pack_is(s["pack"], "E") and c.get("bytes_per_entity_8d") is not None]
    checks = [check("each pack-seed wall time (max)", worst, TARGETS["pack_seed_s"],
                    _le(worst, TARGETS["pack_seed_s"])),
              check("live lib-3 p95 per tick scaled to 35 entities", p95, TARGETS["lib3_p95_ms"],
                    _le(p95, TARGETS["lib3_p95_ms"])),
              check("bytes per entity at end (max)", mem, TARGETS["mem_per_entity"],
                    _le(mem, TARGETS["mem_per_entity"])),
              check("Pack E bytes per entity extrapolated to 8 d", max(e8) if e8 else None,
                    TARGETS["mem_per_entity"], _le(max(e8) if e8 else None,
                                                   TARGETS["mem_per_entity"]))]
    if smoke:
        c = smoke.get("cpu_mem") or smoke
        checks.append(check("smoke total (20 ent, 180x900 s)", c.get("wall_s"),
                            TARGETS["smoke_total_s"], _le(c.get("wall_s"), TARGETS["smoke_total_s"])))
        checks.append(check("smoke lib-3", c.get("lib3_s"), TARGETS["smoke_lib3_s"],
                            _le(c.get("lib3_s"), TARGETS["smoke_lib3_s"])))
    eng: Dict[str, List[float]] = {}
    for c in cm:
        for n, v in (c.get("engine_ms_mean") or {}).items():
            eng.setdefault(n, []).append(v)
    return gate("CPU and memory", worst, TARGETS["pack_seed_s"], checks,
                engine_ms_mean={n: _median(v) for n, v in sorted(eng.items())},
                budgets_ms_at_40=ENGINE_BUDGET_MS)


def gate_robustness(scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rb = [s["robustness"] for s in scores]
    exc = sum(r["exceptions"] for r in rb)
    stale = sum(r["stale"] for r in rb)
    deg = sum(r["pipeline_degraded"] for r in rb if not r["ablated"])
    aborted = [f"{s['pack']}/{s['seed']}: {s['robustness']['aborted']}" for s in scores
               if s["robustness"]["aborted"]]
    checks = [check("engine exceptions (strict)", exc, 0, exc == 0 if rb else None),
              check("stale lib-3 series (> 2 periods)", stale, 0, stale == 0 if rb else None),
              check("pipeline_degraded on clean runs", deg, 0, deg == 0 if rb else None),
              check("aborted runs", len(aborted), 0, not aborted if rb else None)]
    heads = sorted({h for r in rb for h in r["exception_heads"]})[:20]
    return gate("Robustness", exc, 0, checks, exception_heads=heads, aborted=aborted,
                stale_heads=sorted({h for r in rb for h in r["stale_heads"]})[:20])


def design_table(gates: Mapping[str, Dict[str, Any]]) -> str:
    """The design doc's section 3.3 table, regenerated from the gates (gate 16:
    no accuracy claim is written by hand)."""
    lines = ["| # | Gate | Value | Target | Pass |", "|---|---|---|---|---|"]
    for gid, g in gates.items():
        if gid.startswith("16"):
            continue
        p = {True: "yes", False: "no", None: "n/a"}[g.get("pass")]
        lines.append(f"| {gid.split('_')[0]} | {g['name']} | {_fmt(g.get('value'))} | "
                     f"{_fmt(g.get('target'))} | {p} |")
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def compute_gates(scores: Sequence[Dict[str, Any]],
                  feedback: Optional[Sequence[Dict[str, Any]]] = None,
                  ablation: Optional[Mapping[str, Sequence[Dict[str, Any]]]] = None,
                  smoke: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """All 16 gates from per-run scores (full runs only; ablated runs go in
    `ablation`, the feedback runs in `feedback`)."""
    full = [s for s in scores if not s.get("disabled_engines")]
    g: Dict[str, Dict[str, Any]] = {
        "1_detection": gate_detection(full),
        "2_ttd": gate_ttd(full),
        "3_far": gate_far(full),
        "4_legit": gate_legit(full),
        "5_burden": gate_burden(full),
        "6_poisoning": gate_poisoning(full),
        "7_calibration": gate_calibration(full),
        "8_identification": gate_identification(full),
        "9_classes": gate_classes(full),
        "10_explanation": gate_explanation(full),
        "11_portrait": gate_portrait(full),
        "12_feedback": feedback_gate(full, feedback or []),
        "13_ablation": gate_ablation(full, ablation),
        "14_cpu_mem": gate_cpu_mem(full, smoke),
        "15_robustness": gate_robustness(full),
    }
    table = design_table(g)
    g["16_report"] = gate("Design table regenerated from eval_report.json",
                          len(table.splitlines()) - 2, 15,
                          [check("rows generated", len(table.splitlines()) - 2, 15,
                                 len(table.splitlines()) - 2 == 15)], table_md=table)
    return g
