"""Deterministic engine cost model for P15 (budgets, degradation ladder) and
P12 (the CPU cost charged to an arm's utility) — docs/lib3/progressive.md
§6.18.2, §6.19, §16.12.

Why not wall-clock time. Until round 3 both engines read each engine's
measured duration (engine health `duration_ms`). That made two identical runs
decide differently: the same seed scored PG8 0.67 in one run and 0.83 in the
other (pack O seed 1, §16.11.8 item 6) because P12's per-arm costs, and with
them the noise estimate of the switch margin (pstrategy.switch_margin) and the
Hedge weights, moved with the machine's load. The decision inputs are now the
cost of the WORK an engine did, counted from its run:

    ms(run) = scale x (a_e + b_e x units + d_e x events)

  units   the engine's own count of what it processed (health `last_count`:
          rows, nodes fitted, scored events ... - each engine's run() return)
  events  the organisation's events of the tick the run belongs to (evt.batch
          rows over all systems)
  a/b/d   per-engine coefficients, fitted ONCE by non-negative least squares on
          the hourly sums of pack O (5 days, registry progressive_decision,
          one core of the reference machine; scripts/calibrate_costs.py, fit
          table in §16.12). Engines outside the table use DEFAULT_COEF.
  scale   config progressive.budget.cost_scale (default 1): the speed of the
          deployment's machine relative to the reference, measured once.

The model is a price list, not a stopwatch: it keeps cost-awareness (an arm
that does more work is charged more, a P-core that processes more events
uses more of its CPU budget) while two runs with the same inputs make the same
decisions. config progressive.budget.cost_model = 'wall' restores the measured
durations (operations that prefer reacting to the real machine load over
reproducibility). P15 records the measured durations next to the modelled
ones in every mode (model.budget usage.*_wall_*), for operations.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

MODES = ("counted", "wall")
DEFAULT_MODE = "counted"

# engine -> (ms per run, ms per counted unit, ms per org event of the tick)
COEF: Dict[str, Tuple[float, float, float]] = {
    "behavior.attr_registry": (20.64, 0.0141, 0.0049),
    "behavior.attr_select": (78.57, 0.1471, 0),
    "behavior.binding": (24.58, 0.03601, 0.01495),
    "behavior.calibration": (23.41, 0.2597, 0),
    "behavior.conformity": (10.27, 0.1403, 0),
    "behavior.content_bounds": (1.768, 4.413, 0.01818),
    "behavior.explain": (0.01201, 4.269, 2e-05),
    "behavior.facets": (0.1084, 0.473, 0.00037),
    "behavior.fusion": (85.58, 0, 0.03052),
    "behavior.governor": (0, 0.08532, 0.00568),
    "behavior.incident": (4.177, 1.821, 0),
    "behavior.pattern_tree": (5.849, 0.839, 0.04818),
    "behavior.payload_grammar": (0.4052, 0.5523, 0.00018),
    "behavior.resource_governor": (0, 3.108, 0),
    "behavior.risk": (0, 0.0487, 0.00143),
    "behavior.system_profile": (3.192, 0.1656, 0.03826),
    "behavior.time_window": (1.579, 10.09, 0),
    "behavior.views": (0, 1.707, 0.00606),
    "behavior.who_groups": (5.052, 0, 0),
    "behavior.workflow": (1.068, 0.07635, 0),
    "derived.event_context": (0.956, 0.03348, 0),
    "raw.action_token": (0, 0.0061, 0.00716),
    "raw.client_stack": (0.2674, 0.01142, 0),
    "raw.event": (2.561, 0.03954, 0),
}
DEFAULT_COEF: Tuple[float, float, float] = (0.5, 0.01, 0.005)


def mode(budget_cfg: Optional[Mapping[str, Any]]) -> str:
    m = str((budget_cfg or {}).get("cost_model") or DEFAULT_MODE).lower()
    return m if m in MODES else DEFAULT_MODE


def scale(budget_cfg: Optional[Mapping[str, Any]]) -> float:
    try:
        s = float((budget_cfg or {}).get("cost_scale", 1.0) or 1.0)
    except (TypeError, ValueError):
        return 1.0
    return s if math.isfinite(s) and s > 0 else 1.0


def coef(engine: str) -> Tuple[float, float, float]:
    return COEF.get(str(engine), DEFAULT_COEF)


def _num(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) and v > 0 else 0.0


def run_ms(engine: str, units: Any, events: Any, k: float = 1.0) -> float:
    """Modelled duration (ms) of one run of `engine` that processed `units`
    (its own count) at a tick with `events` organisation events."""
    a, b, d = coef(engine)
    return float(k * (a + b * _num(units) + d * _num(events)))


def run_cost_ms(engine: str, rec: Mapping[str, Any], events: Any,
                budget_cfg: Optional[Mapping[str, Any]] = None) -> float:
    """The cost of one engine run as the decisions see it: modelled from its
    health record's count (mode 'counted', the default) or its measured
    duration (mode 'wall')."""
    if mode(budget_cfg) == "wall":
        return _num(rec.get("duration_ms"))
    return run_ms(engine, rec.get("last_count"), events, scale(budget_cfg))
