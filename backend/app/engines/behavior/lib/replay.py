"""Deterministic recompute of stateful detectors (B29 counterfactuals, B28 audits).

Why: an explanation is only faithful if the counterfactual ("without these
features the incident would not have opened") is recomputed through the SAME
stateful statistics (CUSUM bank, MCUSUM, rhythm W, budget sums, identity
CUSUMs), p_from_ring on the stored model.calib snapshot, wHMP and the
decision rule -- not approximated. Every stateful detector therefore
exposes a pure step function and stores its input history (behavior.zr,
behavior.cusum_state, act.* rows) so replay reproduces the live numbers
bit-for-bit from tau-hat.

What lives here (generic, detector-agnostic; the detector maths is in the
owner's accessor module, e.g. m_cp.replay_step / m_rhythm.offhours_step):

  replay(step, state0, inputs, perturb=None, threshold=None) -> ReplayResult
      run a StepFn over time-ordered inputs from a deep copy of state0,
      optionally transforming each input first (the counterfactual), and
      report the score path, the final state and the first alarm.
  counterfactual_features(step, state0, inputs, candidates, threshold, neutral)
      greedy minimal subset of candidates whose neutralisation keeps ONE
      replayed statistic below threshold (forward selection in the given
      order, then backward elimination).
  minimal_set(candidates, flips, max_size=None) -> (subset, n_evals)
      the same greedy search for an arbitrary decision predicate: flips(S)
      is True when neutralising S makes the decision go away (B29 uses it
      with the full recomputed fusion decision, not one statistic).
  load_inputs(store, s, e, names, since, until) -> [(ts, {name: row})]
      aligned vector-ring rows for (since, until] (missing rows are None).

Determinism: nothing here draws randomness; a step that needs a uniform
must seed it (combine.seeded_uniform) from its inputs, so two replays of the
same inputs are identical and a replay of the unperturbed inputs equals the
live statistic (the fidelity check B29 reports).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

StepFn = Callable[[Any, Any], Tuple[Any, float]]     # (state, inputs_at_ts) -> (state, score)
PerturbFn = Callable[[float, Any], Any]              # (ts, inputs) -> perturbed inputs


@dataclass
class ReplayResult:
    ts: List[float]
    scores: List[float]
    final_state: Any
    alarmed_at: Optional[float] = None

    @property
    def max_score(self) -> float:
        """Largest finite score of the path (-inf when none)."""
        fin = [s for s in self.scores if s == s]
        return max(fin) if fin else -math.inf


def replay(step: StepFn, state0: Any, inputs: Sequence[Tuple[float, Any]],
           perturb: Optional[PerturbFn] = None, threshold: Optional[float] = None
           ) -> ReplayResult:
    """Run `step` over (ts, inputs) in ascending ts from state0 (deep-copied),
    optionally transforming each input with perturb(ts, inputs) first.
    alarmed_at = first ts with score >= threshold (if given). Pure. O(len(inputs))."""
    state = copy.deepcopy(state0)
    order = sorted(range(len(inputs)), key=lambda i: float(inputs[i][0]))
    ts_out: List[float] = []
    scores: List[float] = []
    alarmed: Optional[float] = None
    thr = None if threshold is None else float(threshold)
    for i in order:
        t, x = inputs[i]
        t = float(t)
        if perturb is not None:
            x = perturb(t, x)
        state, sc = step(state, x)
        sc = float(sc)
        ts_out.append(t)
        scores.append(sc)
        if alarmed is None and thr is not None and sc == sc and sc >= thr:
            alarmed = t
    return ReplayResult(ts_out, scores, state, alarmed)


def minimal_set(candidates: Sequence[Any], flips: Callable[[Tuple[Any, ...]], bool],
                max_size: Optional[int] = None) -> Tuple[List[Any], int]:
    """Greedy minimal subset S of `candidates` (in their given order, most
    important first) with flips(S) True.

    Forward: add candidates one at a time until flips(S). Backward: drop each
    member (latest added first) whose removal keeps flips(S) True, so the
    result is 1-minimal (no single member can be dropped). flips is called
    with a tuple in candidate order; results are memoised. Returns ([], n)
    when even every candidate (up to max_size) does not flip the decision.
    n = number of distinct flips evaluations.
    """
    cands = list(dict.fromkeys(candidates))          # stable de-duplication
    if max_size is not None:
        cands = cands[:max(0, int(max_size))]
    pos = {c: i for i, c in enumerate(cands)}
    memo: Dict[Tuple[Any, ...], bool] = {}

    def test(sub: Sequence[Any]) -> bool:
        key = tuple(sorted(sub, key=pos.__getitem__))
        if key not in memo:
            memo[key] = bool(flips(key))
        return memo[key]

    chosen: List[Any] = []
    ok = False
    for c in cands:
        chosen.append(c)
        if test(chosen):
            ok = True
            break
    if not ok:
        return [], len(memo)
    for c in list(reversed(chosen[:-1])):
        trial = [x for x in chosen if x != c]
        if trial and test(trial):
            chosen = trial
    return sorted(chosen, key=pos.__getitem__), len(memo)


def counterfactual_features(step: StepFn, state0: Any, inputs: Sequence[Tuple[float, Any]],
                            candidates: Sequence[int], threshold: float,
                            neutral: Callable[[Any, Sequence[int]], Any]) -> List[int]:
    """Greedy minimal subset of feature indices whose neutralisation
    (neutral(inputs, subset), e.g. zr -> 0) keeps the replayed statistic below
    threshold. Returns [] if even all candidates do not suffice."""
    thr = float(threshold)

    def flips(sub: Tuple[int, ...]) -> bool:
        res = replay(step, state0, inputs, perturb=lambda _t, x: neutral(x, list(sub)))
        return res.max_score < thr

    sub, _ = minimal_set(list(candidates), flips)
    return [int(i) for i in sub]


def load_inputs(store: Any, s: str, e: str, names: Sequence[str], since: float, until: float
                ) -> List[Tuple[float, Dict[str, Any]]]:
    """[(ts, {name: vec})] from the store's vector rings for the named series
    in (since, until], one entry per ts of the FIRST name (the replay clock);
    the other names are aligned on exact ts (None where a row is missing).
    Rows are float64 copies."""
    names = list(names)
    if not names:
        return []
    cols: Dict[str, Dict[float, np.ndarray]] = {}
    t0, M0 = store.vec_range(s, e, names[0], float(since), float(until))
    for nm in names[1:]:
        t, M = store.vec_range(s, e, nm, float(since), float(until))
        cols[nm] = {float(a): M[i] for i, a in enumerate(t)}
    out: List[Tuple[float, Dict[str, Any]]] = []
    for i, ts in enumerate(t0):
        ts = float(ts)
        if ts <= float(since):
            continue
        row: Dict[str, Any] = {names[0]: np.asarray(M0[i], dtype=np.float64)}
        for nm in names[1:]:
            v = cols[nm].get(ts)
            row[nm] = None if v is None else np.asarray(v, dtype=np.float64)
        out.append((ts, row))
    return out
