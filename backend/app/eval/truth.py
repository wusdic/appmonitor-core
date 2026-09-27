"""Query helpers over gen.truth (generator.md §5, eval.md LABELS).

gen.truth is a list of plain dicts, one per scenario:
  {scenario_id, pack, system, entities, t_start, t_end,
   label in {malicious, legit_change, system_change},
   expected_detectors, expected_axes, max_ttd | max_ttd_s,
   required_severity | max_allowed_severity, perturbed_features, ...}
plus optional fields: level ('class' | 'system'), class_members, scope='all'
+ background=True (conditions that apply to everyone, entities = []),
link_pairs / negative_control / aliases (L6), chain (T19), looks_like (T9),
windows (T5, L1), onsets (T21, L2, L3), strike_ts (T15).

Only the evaluation reads these: the scoring code in metrics.py and the
runner's snapshot scheduling. Nothing here is ever fed to the pipeline.

Entities in a row may be bare IPs (in the row's system) or 'sys|ip'; every
helper returns fully qualified 'sys|ip' keys.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

LABELS = ("malicious", "legit_change", "system_change")
REQUIRED_FIELDS = ("scenario_id", "pack", "system", "entities", "t_start", "t_end", "label",
                   "expected_detectors", "expected_axes", "perturbed_features")
PRE_S = 3600.0            # control exclusion before t_start
POST_S = 86400.0          # control exclusion after t_end

Interval = Tuple[float, float]


def key(system: str, entity: Any) -> str:
    e = str(entity)
    return e if "|" in e else f"{system}|{e}"


def row_keys(row: Mapping[str, Any]) -> List[str]:
    """Qualified entity keys of one truth row."""
    s = str(row.get("system") or "")
    return [key(s, e) for e in (row.get("entities") or [])]


def base_id(sid: Any) -> str:
    """'T1' from "T1'", 'T6b' stays 'T6b'."""
    s = str(sid).strip().rstrip("'")
    return s.split("-")[0].split("_")[0]


def _f(v: Any, default: float = math.nan) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Row selection
# --------------------------------------------------------------------------- #
def is_background(row: Mapping[str, Any]) -> bool:
    return bool(row.get("background")) or row.get("scope") == "all"


def is_class_row(row: Mapping[str, Any]) -> bool:
    return row.get("level") in ("class", "system") or bool(row.get("class_members"))


def rows(truth: Iterable[Mapping[str, Any]], label: Optional[str] = None,
         include_background: bool = True) -> List[Mapping[str, Any]]:
    out = []
    for r in truth:
        if label is not None and r.get("label") != label:
            continue
        if not include_background and is_background(r):
            continue
        out.append(r)
    return out


def malicious(truth: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return rows(truth, "malicious")


def legitimate(truth: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [r for r in truth if r.get("label") in ("legit_change", "system_change")]


def by_id(truth: Iterable[Mapping[str, Any]], sid: str) -> Optional[Mapping[str, Any]]:
    for r in truth:
        if str(r.get("scenario_id")) == sid:
            return r
    return None


def active_at(truth: Iterable[Mapping[str, Any]], t: float,
              include_background: bool = False) -> List[Mapping[str, Any]]:
    return [r for r in truth if (include_background or not is_background(r))
            and _f(r.get("t_start")) <= t < _f(r.get("t_end"), math.inf)]


def rows_for(truth: Iterable[Mapping[str, Any]], k: str) -> List[Mapping[str, Any]]:
    """Rows whose entities (or aliases / chain / class members) include key k."""
    return [r for r in truth if k in expanded_keys(r)]


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
def scenario_window(row: Mapping[str, Any], grace_s: float = 0.0) -> Interval:
    t0 = _f(row.get("t_start"))
    t1 = _f(row.get("t_end"), t0)
    if math.isnan(t1) or math.isinf(t1):
        t1 = t0
    return t0, t1 + grace_s


def scenario_windows(truth: Iterable[Mapping[str, Any]], grace_s: float = 0.0,
                     include_background: bool = False) -> Dict[str, Interval]:
    return {str(r["scenario_id"]): scenario_window(r, grace_s) for r in truth
            if include_background or not is_background(r)}


def active_windows(row: Mapping[str, Any]) -> List[Interval]:
    """The sub-windows a scenario is actually active in (T5 slots, L1 days,
    L10 outage + degradation) or its whole [t_start, t_end)."""
    for fld in ("windows",):
        w = row.get(fld)
        if w:
            return [(float(a), float(b)) for a, b in w]
    if row.get("outage") and row.get("degrade"):
        return [tuple(map(float, row["outage"])), tuple(map(float, row["degrade"]))]  # type: ignore
    return [scenario_window(row)]


def deadline_s(row: Mapping[str, Any], dt: float) -> Optional[float]:
    """Deadline in seconds from t_start: max_ttd_s, or max_ttd as seconds /
    '<n> ticks' / '<n>h' / '<n>min' / '<n>d'."""
    if row.get("max_ttd_s") is not None:
        return _f(row["max_ttd_s"])
    v = row.get("max_ttd")
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    num = ""
    i = 0
    while i < len(s) and (s[i].isdigit() or s[i] == "."):
        num += s[i]
        i += 1
    unit = s[i:].strip()
    mult = {"": 1.0, "s": 1.0, "min": 60.0, "m": 60.0, "h": 3600.0, "d": 86400.0,
            "tick": dt, "ticks": dt}.get(unit)
    if not num or mult is None:
        return None
    return float(num) * mult


# --------------------------------------------------------------------------- #
# Aliases, chains, classes
# --------------------------------------------------------------------------- #
def alias_sets(truth: Iterable[Mapping[str, Any]]) -> List[Set[str]]:
    """Ground-truth identity sets: continuity pairs (L6 renumbering) and actor
    chains (T19 IP hopping), as sets of 'sys|ip' keys. Overlapping sets are
    merged."""
    groups: List[Set[str]] = []
    for r in truth:
        s = str(r.get("system") or "")
        for fld in ("link_pairs", "aliases"):
            for grp in r.get(fld) or []:
                groups.append({key(s, e) for e in grp})
        if r.get("chain"):
            groups.append({key(s, e) for e in r["chain"]})
    merged: List[Set[str]] = []
    for g in groups:
        g = set(g)
        keep = []
        for m in merged:
            if m & g:
                g |= m
            else:
                keep.append(m)
        merged = keep + [g]
    return [m for m in merged if len(m) >= 2]


def alias_map(truth: Iterable[Mapping[str, Any]]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for g in alias_sets(truth):
        for k in g:
            out[k] = set(g) - {k}
    return out


def class_members(row: Mapping[str, Any]) -> Set[str]:
    """True class members of a class / system scenario (ground truth, not
    the clustering's view); empty for entity scenarios."""
    s = str(row.get("system") or "")
    mem = row.get("class_members")
    if mem:
        return {key(s, e) for e in mem}
    if row.get("level") in ("class", "system"):
        return set(row_keys(row))
    return set()


def expanded_keys(row: Mapping[str, Any]) -> Set[str]:
    """Scenario entity keys + negative controls + chain (not class members)."""
    s = str(row.get("system") or "")
    out = set(row_keys(row))
    for fld in ("chain", "negative_control"):
        out |= {key(s, e) for e in row.get(fld) or []}
    for fld in ("link_pairs", "aliases"):
        for grp in row.get(fld) or []:
            out |= {key(s, e) for e in grp}
    return out


def scenario_entities(truth: Iterable[Mapping[str, Any]]) -> Set[str]:
    """Every entity that belongs to some scenario of the pack."""
    out: Set[str] = set()
    for r in truth:
        out |= expanded_keys(r)
    return out


def control_entities(truth: Iterable[Mapping[str, Any]], population: Iterable[str]) -> List[str]:
    """Population keys with no scenario in the pack (eval.md LABELS)."""
    scen = scenario_entities(list(truth))
    return sorted(k for k in population if k not in scen)


def _merge(iv: Iterable[Interval]) -> List[Interval]:
    out: List[Interval] = []
    for a, b in sorted(iv):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def exclusion_windows(truth: Iterable[Mapping[str, Any]], k: str,
                      class_of: Optional[Mapping[str, Iterable[str]]] = None) -> List[Interval]:
    """Windows [t_start - 1 h, t_end + 24 h] of every class / system scenario
    whose class includes control entity k. Class membership is the truth's
    class_members, or, when `class_of` is given ({key: class ids}), any
    shared class with the scenario's entities."""
    truth = list(truth)
    out = []
    mine = set(class_of.get(k, ())) if class_of else set()
    for r in truth:
        if not is_class_row(r) or is_background(r):
            continue
        member = k in class_members(r)
        if not member and class_of:
            theirs: Set[str] = set()
            for e in row_keys(r):
                theirs |= set(class_of.get(e, ()))
            member = bool(mine & theirs)
        if not member and r.get("level") == "system":
            member = k.partition("|")[0] == str(r.get("system"))
        if member:
            t0, t1 = scenario_window(r)
            out.append((t0 - PRE_S, t1 + POST_S))
    return _merge(out)


def control_windows(truth: Iterable[Mapping[str, Any]], population: Iterable[str],
                    t0: float, t1: float,
                    class_of: Optional[Mapping[str, Iterable[str]]] = None,
                    first_seen: Optional[Mapping[str, float]] = None
                    ) -> Dict[str, List[Interval]]:
    """Exposure intervals of each control entity inside [t0, t1]: the span
    minus its class-scenario exclusion windows (and before first_seen)."""
    truth = list(truth)
    out: Dict[str, List[Interval]] = {}
    for k in control_entities(truth, population):
        lo = max(t0, float((first_seen or {}).get(k, t0)))
        cur = [(lo, t1)] if t1 > lo else []
        for a, b in exclusion_windows(truth, k, class_of):
            nxt = []
            for x, y in cur:
                if b <= x or a >= y:
                    nxt.append((x, y))
                    continue
                if a > x:
                    nxt.append((x, a))
                if b < y:
                    nxt.append((b, y))
            cur = nxt
        out[k] = cur
    return out


def exposure_s(windows: Mapping[str, Sequence[Interval]]) -> float:
    return float(sum(b - a for iv in windows.values() for a, b in iv))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(truth: Sequence[Mapping[str, Any]]) -> List[str]:
    """Problems with a truth list (empty when it is complete, §5)."""
    errs: List[str] = []
    seen: Dict[str, str] = {}
    for r in truth:
        sid = str(r.get("scenario_id"))
        for fld in REQUIRED_FIELDS:
            if fld not in r:
                errs.append(f"{sid}: missing {fld}")
        if r.get("label") not in LABELS:
            errs.append(f"{sid}: bad label {r.get('label')!r}")
        t0, t1 = _f(r.get("t_start")), _f(r.get("t_end"))
        if not (math.isfinite(t0) and math.isfinite(t1) and t1 > t0):
            errs.append(f"{sid}: bad window {t0}..{t1}")
        if r.get("label") == "malicious":
            if not r.get("required_severity"):
                errs.append(f"{sid}: no required_severity")
            if r.get("max_ttd") is None and r.get("max_ttd_s") is None:
                errs.append(f"{sid}: no max_ttd")
            if not (r.get("expected_detectors") or r.get("expected_axes")):
                errs.append(f"{sid}: nothing expected")
            if not r.get("perturbed_features"):
                errs.append(f"{sid}: no perturbed_features")
            if not r.get("entities"):
                errs.append(f"{sid}: no entities")
        elif not r.get("max_allowed_severity"):
            errs.append(f"{sid}: no max_allowed_severity")
        if sid in seen:
            errs.append(f"{sid}: duplicate scenario id")
        seen[sid] = sid
        for k in row_keys(r):
            other = seen.get("@" + k)
            if other is not None:
                errs.append(f"{sid}: entity {k} already in {other} (one scenario per entity)")
            seen["@" + k] = sid
    return errs
