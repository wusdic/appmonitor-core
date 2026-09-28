"""Shared write/read convention for detector outputs (contract B, J).

Up to ~20 engines each own a few columns of the same per-tick vectors
(behavior.score / behavior.pm are aligned to detectors.DETECTORS) and a few
keys of the same per-tick dicts (behavior.axes, behavior.acc_alarm,
behavior.degraded). They all write in the same tick, so writes must MERGE
rather than append: every detector engine goes through `write_scores`, which
uses the store's same-tick upserts. Columns nobody wrote stay NaN, meaning
"not scored this tick" — never p = 1.

Like gating/replay/featcache, this module takes the store as an argument; it
holds no state and imports no engine.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from .detectors import DETECTOR_INDEX, DETECTORS, N_DETECTORS

SCORE = "behavior.score"        # raw detector scores (higher = more anomalous)
PM = "behavior.pm"              # model p-values (exposure-exact), optional per detector
P = "behavior.p"                # calibrated p-values — written only by B24
AXES = "behavior.axes"          # {detector: [axis, ...]}
ACC_ALARM = "behavior.acc_alarm"  # {detector: 0|1} for accumulators
DEGRADED = "behavior.degraded"  # {detector: cause}

# behavior.degraded causes (contract M; the detector-health panel shows the
# share of scored entities whose detector ran degraded at their last tick).
# A detector that is NaN because an input is missing writes stale: /
# producer_error: / unscorable:. A detector that still scores, but on a
# weaker reference than its model assumes, writes one of the other three:
# its p is valid (B24 calibrates it) but less informative, and B25 keeps it
# (a family is degraded only when it also has no valid p).
STALE = "stale"                        # stale:<input series>
PRODUCER_ERROR = "producer_error"      # producer_error:<engine>
UNSCORABLE = "unscorable"              # unscorable:<model>
INSUFFICIENT_SUPPORT = "insufficient_support"   # too little own history / data
FALLBACK = "fallback"                  # fallback:<tier> (class / system / hyper prior)
PROVISIONAL = "provisional"            # provisional:<what> (e.g. q_transfer)
CAUSES = (STALE, PRODUCER_ERROR, UNSCORABLE, INSUFFICIENT_SUPPORT, FALLBACK, PROVISIONAL)


def cause(kind: str, detail: Optional[str] = None) -> str:
    """A behavior.degraded cause string 'kind' or 'kind:detail' (kind in CAUSES)."""
    if kind not in CAUSES:
        raise ValueError(f"unknown degraded cause {kind!r} (expected one of {CAUSES})")
    return kind if not detail else f"{kind}:{detail}"


def write_degraded(store, system: str, entity: str, ts: float,
                   degraded: Mapping[str, str], window_s: Optional[int] = None) -> None:
    """Merge {detector: cause} into behavior.degraded at ts (same-tick upsert:
    several engines, and several passes of one engine, may add detectors)."""
    if not degraded:
        return
    for name in degraded:
        if name not in DETECTOR_INDEX:
            raise KeyError(f"unknown detector {name!r}")
    store.upsert_dict(system, entity, DEGRADED, ts, {k: str(v) for k, v in degraded.items()},
                      window_s or 0)

_VIRTUAL_PREFIX = {SCORE: "behavior.score.", PM: "behavior.pm.", P: "behavior.p."}


def ensure_registered(store) -> None:
    """Expose behavior.score.<d>, behavior.pm.<d>, behavior.p.<d> as virtual
    scalar views (idempotent; registration is global to the store)."""
    for vec, prefix in _VIRTUAL_PREFIX.items():
        if store.vec_columns(vec) is None:
            store.register_vector_names(vec, DETECTORS, prefix)


def _cols(values: Mapping[str, Optional[float]]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for name, v in values.items():
        if name not in DETECTOR_INDEX:
            raise KeyError(f"unknown detector {name!r}")
        out[DETECTOR_INDEX[name]] = np.nan if v is None else float(v)
    return out


def write_scores(store, system: str, entity: str, ts: float,
                 scores: Mapping[str, Optional[float]],
                 pm: Optional[Mapping[str, Optional[float]]] = None,
                 axes: Optional[Mapping[str, Iterable[str]]] = None,
                 acc_alarm: Optional[Mapping[str, int]] = None,
                 degraded: Optional[Mapping[str, str]] = None,
                 window_s: Optional[int] = None) -> None:
    """Merge one engine's detector outputs into the tick's shared rows.

    scores/pm: {detector: value}; None or NaN means 'unscored/degraded'.
    axes: {detector: [axis]} for detectors that fired or contributed.
    acc_alarm: {detector: 0|1} for accumulator detectors.
    degraded: {detector: cause} when an input was stale or a producer failed.
    """
    ensure_registered(store)
    if scores:
        store.upsert_vec(system, entity, SCORE, ts, _cols(scores), N_DETECTORS, window_s)
    if pm:
        store.upsert_vec(system, entity, PM, ts, _cols(pm), N_DETECTORS, window_s)
    if axes:
        store.upsert_dict(system, entity, AXES, ts, {k: list(v) for k, v in axes.items()},
                          window_s or 0)
    if acc_alarm:
        store.upsert_dict(system, entity, ACC_ALARM, ts, {k: int(v) for k, v in acc_alarm.items()},
                          window_s or 0)
    if degraded:
        write_degraded(store, system, entity, ts, degraded, window_s)


def write_pvalues(store, system: str, entity: str, ts: float,
                  pvals: Mapping[str, Optional[float]], window_s: Optional[int] = None) -> None:
    """B24 only: merge calibrated p-values into behavior.p at ts."""
    ensure_registered(store)
    store.upsert_vec(system, entity, P, ts, _cols(pvals), N_DETECTORS, window_s)


def read_row(store, system: str, entity: str, name: str, ts: float) -> Dict[str, float]:
    """{detector: value} of vector `name` (SCORE/PM/P) at exactly ts; NaN
    entries are omitted. Empty dict when no row exists at ts."""
    row = store.vec_at(system, entity, name, ts)
    if row is None:
        return {}
    return {d: float(v) for d, v in zip(DETECTORS, row) if not math.isnan(float(v))}


def read_array(store, system: str, entity: str, name: str, ts: float) -> np.ndarray:
    """Full aligned float64 row at ts (all-NaN when absent)."""
    row = store.vec_at(system, entity, name, ts)
    if row is None:
        return np.full(N_DETECTORS, np.nan)
    return np.asarray(row, dtype=np.float64)


def read_dict(store, system: str, entity: str, name: str, ts: float) -> Dict[str, Any]:
    """Dict series (AXES / ACC_ALARM / DEGRADED) at exactly ts, else {}."""
    m = store.latest_derived(system, entity, name)
    if m is not None and m.ts == float(ts) and isinstance(m.value, dict):
        return dict(m.value)
    for m in reversed(store.derived_tail(system, entity, name, 4)):
        if m.ts == float(ts) and isinstance(m.value, dict):
            return dict(m.value)
    return {}


def scored_detectors(store, system: str, entity: str, ts: float) -> List[str]:
    return list(read_row(store, system, entity, SCORE, ts).keys())
