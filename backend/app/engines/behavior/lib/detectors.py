"""Detector registry (contract J, architecture section 4).

Every detector score / p-value vector (behavior.score, behavior.pm,
behavior.p) is aligned to DETECTORS, so the order here is part of the store
contract and must never change (append only). Fusion groups detectors by
family, the evidence CUSUM runs only over instantaneous ('inst') detectors,
and each accumulator ('acc') carries its own wall-clock false-alarm budget.

budget_per_day: the null alarm budget per entity-day (per class-day for class
detectors) that the accumulator's time-based threshold is tuned to. It is the
architecture section 4 path budget split evenly over the path's detectors
(eval gate 7 enforces the path totals in BUDGET_PATHS). Instantaneous
detectors have no own budget: they share the single-tick and evidence-CUSUM
paths.
"""
from __future__ import annotations

import math

from typing import Dict, List, Optional

import numpy as np

DETECTORS: List[str] = [
    "marg_int", "marg_shape", "peer", "t2", "spe", "offhours", "silence", "novelty",
    "novelty_rate", "jsd", "client", "seq", "dwell", "timing", "beacon", "budget_vol",
    "budget_exfil", "budget_breadth", "cusum", "mcusum", "bocpd", "creep", "identity",
    "class_int", "class_shape", "class_rhythm", "class_novel", "class_coherence",
    "mixture", "session", "cross_system",
]
N_DETECTORS: int = len(DETECTORS)
DETECTOR_INDEX: Dict[str, int] = {d: i for i, d in enumerate(DETECTORS)}

FAMILIES: List[str] = ["intensity", "shape", "peer", "temporal", "categorical", "breadth",
                       "exfil", "sequence", "identity", "change", "c2", "xsys"]

FAMILY_DEFAULT_AXES: Dict[str, List[str]] = {
    "intensity": ["volume"],
    "shape": ["shape"],
    "peer": ["peer"],            # engines refine to the feature groups involved
    "temporal": ["temporal"],
    "categorical": ["categorical"],  # refined to exfil / privilege per value
    "breadth": ["breadth"],
    "exfil": ["exfil"],
    "sequence": ["sequence"],    # refined to credential for auth token families
    "identity": ["identity"],
    "change": ["change"],        # engines refine to the contributing feature groups
    "c2": ["c2"],
    "xsys": ["discovery"],
}

# Null alarm budget per entity-day by decision path (architecture section 4).
BUDGET_PATHS: Dict[str, float] = {
    "single_tick": 0.03,
    "evidence_cusum": 0.03,
    "change": 0.02,
    "rhythm": 0.015,
    "budget": 0.01,
    "identity_cusum": 0.01,
    "temporal_categorical": 0.02,   # beacon, timing, jsd, novelty_rate
    "class": 0.01,                  # class accumulators, per class-day
}

# name: (family, kind, owner engine, budget_path (acc only), p2)
_TABLE = {
    "marg_int": ("intensity", "inst", "B04", None, False),
    "marg_shape": ("shape", "inst", "B04", None, False),
    "peer": ("peer", "inst", "B04", None, False),
    "t2": ("intensity", "inst", "B06", None, False),
    "spe": ("shape", "inst", "B06", None, False),
    "offhours": ("temporal", "acc", "B07", "rhythm", False),
    "silence": ("temporal", "acc", "B07", "rhythm", False),
    "novelty": ("categorical", "inst", "B08", None, False),
    "novelty_rate": ("breadth", "acc", "B08", "temporal_categorical", False),
    "jsd": ("categorical", "acc", "B08", "temporal_categorical", False),
    "client": ("identity", "inst", "B09", None, False),
    "seq": ("sequence", "inst", "B10", None, False),
    "dwell": ("sequence", "inst", "B10", None, False),
    "timing": ("temporal", "acc", "B11", "temporal_categorical", False),
    "beacon": ("c2", "acc", "B12", "temporal_categorical", False),
    "budget_vol": ("intensity", "acc", "B13", "budget", False),
    "budget_exfil": ("exfil", "acc", "B13", "budget", False),
    "budget_breadth": ("breadth", "acc", "B13", "budget", False),
    "cusum": ("change", "acc", "B14", "change", False),
    "mcusum": ("change", "acc", "B14", "change", False),
    "bocpd": ("change", "acc", "B14", "change", False),
    "creep": ("change", "acc", "B14", "change", False),
    "identity": ("identity", "inst", "B16", None, False),
    "class_int": ("intensity", "inst", "B18", None, False),
    "class_shape": ("shape", "inst", "B18", None, False),
    "class_rhythm": ("temporal", "acc", "B18", "class", False),
    "class_novel": ("categorical", "acc", "B18", "class", False),
    "class_coherence": ("peer", "inst", "B18", None, False),
    "mixture": ("shape", "inst", "B19", None, True),
    "session": ("sequence", "inst", "B20", None, True),
    "cross_system": ("xsys", "inst", "B21", None, True),
}

_PATH_SIZE: Dict[str, int] = {}
for _d, (_f, _k, _o, _p, _p2) in _TABLE.items():
    if _p is not None:
        _PATH_SIZE[_p] = _PATH_SIZE.get(_p, 0) + 1

CLASS_DETECTORS = frozenset({"class_int", "class_shape", "class_rhythm", "class_novel",
                             "class_coherence"})

DETECTOR_INFO: Dict[str, Dict[str, object]] = {}
for _d in DETECTORS:
    _f, _k, _o, _p, _p2 = _TABLE[_d]
    info: Dict[str, object] = {
        "family": _f,
        "kind": _k,
        "axes": list(FAMILY_DEFAULT_AXES[_f]),
        "owner": _o,
        "p2": _p2,
        # Mondrian strata used by B24; identity uses (daypart, regime tercile).
        "strata": "daypart_regime" if _d == "identity" else "daypart_cc",
        # class detectors score class:<id> pseudo-entities, not real entities
        "level": "class" if _d in CLASS_DETECTORS else "entity",
    }
    if _k == "acc":
        info["budget_path"] = _p
        info["budget_per_day"] = BUDGET_PATHS[_p] / _PATH_SIZE[_p]
    DETECTOR_INFO[_d] = info

INSTANT_DETECTORS: List[str] = [d for d in DETECTORS if DETECTOR_INFO[d]["kind"] == "inst"]
ACC_DETECTORS: List[str] = [d for d in DETECTORS if DETECTOR_INFO[d]["kind"] == "acc"]
P2_DETECTORS: List[str] = [d for d in DETECTORS if DETECTOR_INFO[d]["p2"]]
INSTANT_IDX: List[int] = [DETECTOR_INDEX[d] for d in INSTANT_DETECTORS]
ACC_IDX: List[int] = [DETECTOR_INDEX[d] for d in ACC_DETECTORS]


def family_members(family: str, kind: Optional[str] = None) -> List[str]:
    """Detectors of `family` in DETECTORS order, optionally only 'inst' or 'acc'."""
    if family not in FAMILY_DEFAULT_AXES:
        raise KeyError(family)
    return [d for d in DETECTORS if DETECTOR_INFO[d]["family"] == family
            and (kind is None or DETECTOR_INFO[d]["kind"] == kind)]


def family_of(detector: str) -> str:
    return str(DETECTOR_INFO[detector]["family"])


def is_instant(detector: str) -> bool:
    return DETECTOR_INFO[detector]["kind"] == "inst"


# Families that contain at least one instantaneous detector (p_inst is taken
# over the instantaneous members of these families only).
INSTANT_FAMILIES: List[str] = [f for f in FAMILIES if family_members(f, "inst")]

FAMILY_MEMBERS: Dict[str, List[str]] = {f: family_members(f) for f in FAMILIES}
FAMILY_IDX: Dict[str, List[int]] = {f: [DETECTOR_INDEX[d] for d in m]
                                    for f, m in FAMILY_MEMBERS.items()}


def detectors_of(owner: str) -> List[str]:
    """Detectors whose scores engine `owner` (e.g. 'B14') writes."""
    return [d for d in DETECTORS if DETECTOR_INFO[d]["owner"] == owner]


def new_score_vector() -> np.ndarray:
    """A fresh behavior.score / pm / p row: float64[31] of NaN (NaN = unscored)."""
    return np.full(N_DETECTORS, np.nan)


def acc_level(p: float, detector: str, dt_s: float) -> float:
    """Shared accumulator level L ~ S/h from the detector's CALIBRATED p
    (integration: B27's quiet test and B28's SUSPECT / trust tests use the
    same scale). A CUSUM's stationary tail is P(S >= x) ~ exp(-theta x) with
    theta h ~ ln ARL (ARL in ticks from the detector's wall-clock budget), so
    L = ln(1/p) / ln(ARL_ticks); L >= 1 is the alarm level. The raw
    m_cp.level (max S/h over B14's 48 charts) is NOT this scale: on a null it
    sits >= 1/4 on most ticks. NaN p (or ARL <= 1 tick) -> NaN."""
    if not (p == p) or not dt_s > 0:
        return math.nan
    arl = arl_days(detector) * 86400.0 / float(dt_s)
    if arl <= 1.0:
        return math.nan
    pv = min(1.0, max(1e-300, float(p)))
    return -math.log(pv) / math.log(arl)


def arl_days(detector: str) -> float:
    """Wall-clock ARL target (days) of an accumulator = 1 / budget_per_day."""
    info = DETECTOR_INFO[detector]
    if info["kind"] != "acc":
        raise ValueError(f"{detector} is instantaneous; it has no own budget")
    return 1.0 / float(info["budget_per_day"])  # type: ignore[arg-type]
