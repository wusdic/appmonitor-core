"""Deterministic recompute of stateful detectors (B29 counterfactuals, B28 audits).

STATUS: minimal contract stub; implemented in a later wave.

Why: an explanation is only faithful if the counterfactual ("without these
features the incident would not have opened") is recomputed through the SAME
stateful statistics (CUSUM bank, MCUSUM, rhythm W, budget sums, identity
CUSUMs), p_from_ring on the stored model.calib snapshot, wHMP and the
decision rule -- not approximated. Every stateful detector therefore
exposes a pure step function and stores its input history (behavior.zr,
behavior.cusum_state, act.* rows) so replay reproduces the live numbers
bit-for-bit from tau-hat.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

StepFn = Callable[[Any, Any], Tuple[Any, float]]     # (state, inputs_at_ts) -> (state, score)
PerturbFn = Callable[[float, Any], Any]              # (ts, inputs) -> perturbed inputs


@dataclass
class ReplayResult:
    ts: List[float]
    scores: List[float]
    final_state: Any
    alarmed_at: Optional[float] = None


def replay(step: StepFn, state0: Any, inputs: Sequence[Tuple[float, Any]],
           perturb: Optional[PerturbFn] = None, threshold: Optional[float] = None
           ) -> ReplayResult:
    """Run `step` over (ts, inputs) in ascending ts from state0 (deep-copied),
    optionally transforming each input with perturb(ts, inputs) first.
    alarmed_at = first ts with score >= threshold (if given). Pure. O(len(inputs))."""
    raise NotImplementedError("replay.replay")


def counterfactual_features(step: StepFn, state0: Any, inputs: Sequence[Tuple[float, Any]],
                            candidates: Sequence[int], threshold: float,
                            neutral: Callable[[Any, Sequence[int]], Any]) -> List[int]:
    """Greedy minimal subset of feature indices whose neutralisation
    (neutral(inputs, subset), e.g. zr -> 0) keeps the replayed statistic below
    threshold. Returns [] if even all candidates do not suffice."""
    raise NotImplementedError("replay.counterfactual_features")


def load_inputs(store: Any, s: str, e: str, names: Sequence[str], since: float, until: float
                ) -> List[Tuple[float, Dict[str, Any]]]:
    """[(ts, {name: vec})] from store.vec_since for the named vector series in (since, until]."""
    raise NotImplementedError("replay.load_inputs")
