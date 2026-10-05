HARNESS

- Code lives in backend/app/eval/{packs.py, truth.py, metrics.py, runner.py}.
- scripts/evaluate.py runs 5 seeds × 5 packs as parallel processes. It writes eval_report.json and eval_report.html; the HTML is served at GET /api/eval/report.
- tests/test_lib3_accuracy.py (marked slow, nightly) enforces the gates below.
- tests/engines/test_<engine>.py holds the per-engine unit tests from the engine specs.
  - tests/helpers.py is extended with add_vec_rows, add_obs_tick, put_model, make_tctx, set_trust and run_engine(strict=True).
  - The whole per-engine suite must run in under 20 s.
- All runs use ctx.config['strict'] = True. Any engine exception, or any produced series left stale for more than 2 of its periods, fails the run.

LABELS

- gen.truth is the only source of ground truth, and the scoring code never reads it.
- A detector or axis counts as a hit only through the incident's p_by_detector and axes, or through discrete events on the scenario entity. The scenario entity includes its continuity aliases, its actor chain and, for class scenarios, its class key.
- Control entities are those with no scenario in the pack. Their windows exclude [t_start − 1 h, t_end + 24 h] of any scenario whose class includes them.

METRICS AND ACCEPTANCE TARGETS

Values are medians over 5 seeds, with bootstrap 95% confidence intervals reported.

1. Detection
   - True positive: an incident that meets the required severity, opens within [t_start, t_end + max(4 ticks, 1 h)], and whose axes or detectors include at least one expected one.
   - True positive by escalation (lead decision, round 4): an attack on an entity whose incident is already open at t_start is folded into that incident by B27 (one incident per episode) instead of opening its own. It counts as a detection if, within [t_start, t_end + max(4 ticks, 1 h)], that incident's severity rises by at least one level above its level at t_start to at least max(MEDIUM, required severity), it gains at least one of the scenario's expected AXES it did not carry at t_start, and an `escalate` notification of it is emitted at or after that point. TTD is taken from that notification. A reopening inside the window is an opening, not an escalation.
   - The opening-only rule (the rule before round 4) is kept as a reported secondary metric: `recall_open` / "overall threat recall, opening-only rule", and per scenario `detected_open`, `within_deadline_open`, `ttd_s_open`; `detected_by` says which rule produced the TTD (the earlier of the two). FAR (gate 3) still counts per incident id.
   - Loud scenarios (T1, T6, T6b, T9, T11, T12): recall 1.00 across all seeds, TTD ≤ 2 ticks at 900 s (≤ 30 min wall).
   - Subtle scenarios (T2, T3, T4, T4b, T5, T7, T8, T9b, T10, T13–T19, T21): recall ≥ 0.90 pooled, each within its own deadline.
   - Overall threat recall ≥ 0.95.
   - Range-based precision, recall and F1 (Tatbul 2018) are reported. Point-adjusted F1 is shown for reference only.

2. Time to detect: median and p90 in ticks and in wall time for every scenario. In Pack E, the wall-clock TTD at 60 s must be ≤ the TTD at 900 s plus 15 min.

3. False alarms on control entities
   - Incidents ≥ LOW: ≤ 0.2 per entity-day.
   - Incidents ≥ MEDIUM: ≤ 0.05 per entity-day.
   - HIGH or above: ≤ 1 in total across 5 seeds × all packs. The design expectation is ≈ 0.12 over ≈ 1225 clean entity-days.
   - CRITICAL: 0.
   - Bursty and steady entities stay within 2× of each other.
   - Cadence invariance: FAR ≥ LOW in Pack E at 60 s is within [0.5, 2]× the Pack A FAR at 900 s.

4. Legitimate scenarios L1–L16: at least 95% of seed × scenario runs stay at or below the allowed severity, and affected entities and classes keep max risk < 30.

5. Alert burden
   - Notifications per TP incident ≤ 3 (open + escalate notifications of the TP incident's episode; for a TP by escalation only those sent from t_start on, i.e. not the notifications of its earlier FP life). The mean over TPs opened in the window alone is reported as a secondary check.
   - Incidents per episode = 1 (for example T2 over 5 days).
   - Class-wide legitimate changes: at most 1 class incident per event.

6. Poisoning and reversibility
   - Measured at t_end + max(8 ticks, 2 h), after any governor rollback:
     - the current-anchor predictive mean of the perturbed features is within 0.3σ15 of pre-onset (0.3 log1p for T2);
     - the reference anchor is within 0.1σ15;
     - golden is unchanged;
     - no model.control version bump happens inside a malicious window.
   - Rollback correctness: for every rollback, the replayed baseline equals an offline fit on rows ≤ τ̂ within 1e-6.
   - After an attack ends, the incident closes within max(8 ticks, 2 h), and null KS D ≤ 0.05 within 96 ticks.
   - Legitimate accepts:
     - L5 is accepted within 3 days after first SUSPECT;
     - L9 within 4 days;
     - an L-type single-entity intensity change within 1.5 days;
     - no entity stays DRIFTING for more than 14 days without entering the label queue.

7. Calibration (clean control ticks)
   - KS D of randomized behavior.p.<d> against U(0,1) ≤ 0.05 per detector, pooled over entities and strata.
   - Empirical exceedance at e_day ∈ {0.03, 3e-3} within [0.5, 2]× nominal, for each cadence class.
   - Evidence-CUSUM realised alarm rate ≤ 0.045 per entity-day, i.e. an interval ≥ 22 entity-days against the 33-day design.
   - Each accumulator family's realised rate ≤ 2× its budget in architecture_md §4.
   - The ACAT-masking regression case must not occur: no incident with p_family > 0.5 while any member has p < 1e-4.

8. Identification (B15 blocked CV and the B16 replay; absolute representation)
   - Individuated personas: window top-1 (K = 4) ≥ 0.95, per-tick top-1 ≥ 0.85 at 900 s, and EER_hard ≤ 0.05 for ≥ 90% of personas.
   - Twins: EER_hard > 0.2, each listed as confusable_with the other.
   - Spearman correlation of separability against CV recall ≥ 0.8. CV recall is graded (round 4): B15's median held-out margin log L(own) − max log L(other) per window (`profile.extra.identity.margin`; recall@1 is the share of windows with margin > 0 and is 1.0 for every individuated persona of packs A and B, which left the correlation undefined), over the individuated personas and the twins (the designed low-separability pair); recall@1 is the fallback grade when no margin is published.
   - T9 looks_like correct in 100% of seeds; T9b unknown_identity in ≥ 90%.
   - L6 link precision 1.0 (the negative control is never linked) and recall ≥ 0.95.
   - T19 actor chain recovered in ≥ 90% of seeds.

9. Classes
   - Level-1 role ARI against true archetypes ≥ 0.90.
   - ≤ 10% of personas end up as noise or unique at level 1.
   - Level 2 separates the individual personas: sub-class purity ≥ 0.8.
   - ARI between consecutive refits ≥ 0.95, and zero ID churn over 10 refits with unchanged membership.
   - L8 typing prob ≥ 0.8 within 3 ticks.
   - T21 class detection in ≥ 90% of seeds; L1, L2 and L3 class incidents ≤ LOW.

10. Explanation
    - hit@3 ≥ 0.8 against perturbed_features. The whole-entity marker 'entity' (L6 renumbering, L8 new employee) is not a feature an attribution can rank: such rows have no hit@3 (round 4); their TP incidents are counted apart ("TP incidents without a rankable feature").
    - Counterfactual validity ≥ 0.9, measured on the full recomputed decision via replay; the scope includes stateful detectors.
    - A natural-unit range is present in 100% of incidents.

11. Portrait
    - Typical-hours Jaccard against the persona window ≥ 0.8.
    - Top-5 template recall ≥ 0.8.
    - p5–p95 coverage of held-out ticks within [0.85, 0.95].
    - Class portraits exist for 100% of role, static and pool classes.

12. Feedback: a simulated analyst with 5% label noise and 5 labels per day (the queue, round 4: each pick draws one unlabelled incident active in the last 24 h and not suppressed by a policy, with probability proportional to 2^rank of its severity — LOW 1, MEDIUM 2, HIGH 4, CRITICAL 8 — and an entity is reviewed at most once a day; the former newest-first pick labelled the threat updated last almost every time) cuts control incidents ≥ LOW by ≥ 50% after 20 labels, with recall dropping by ≤ 0.02. Suppression escape behaves as in unit test B23. The analyst dismisses fp verdicts with scope = pattern (B23 builds suppression policies only from widened scopes) and labels the rest with scope = this; the cut compares each feedback run with the full run of the same (pack, seed). The cut counts control incidents ≥ LOW that were ever NOTIFYING (an open / escalate notification, else a history point with status open / acked): an incident a pattern policy suppresses from its first tick is the suppression working and never reaches the analyst (round 4); the cut over all incidents, suppressed ones included, is reported as a secondary check. Ablation deltas (gate 13) are likewise taken against the full runs of the same (pack, seed).

13. Ablation
    - Δrecall and ΔFAR are reported per scenario with each engine disabled.
    - Every P0/P1 engine is the sole or main detector of at least one scenario; B18 must be the sole detector of T21.
    - P2 engines (B19–B22) are enabled by default only if they improve at least one gate without breaking any other.

14. CPU and memory
    - Smoke run (20 entities, 180 × 900 s): total ≤ 30 s, of which lib-3 ≤ 12 s.
    - Live lib-3 p95 ≤ 80 ms per tick at 35 entities.
    - Per-engine budgets as specified.
    - Each pack-seed ≤ 6 min on one core.
    - store.memory_report() at steady state: ≤ 12 MB per entity, extrapolated to 8 days at 60 s from Pack E once retention has saturated.

15. Robustness
    - 0 engine exceptions in strict mode.
    - pipeline_degraded is never emitted on clean runs.
    - Fault injection: disabling B04 degrades its family (NaN), not p = 1, and FAR does not rise.

16. The design doc's §3.3 table is regenerated from eval_report.json. No accuracy claim is written by hand.

PROGRESSIVE CORE (PG1–PG11): MEASUREMENT SEMANTICS (round 4)

The PG gates are defined in progressive.md §12 and scored by backend/app/eval/pmetrics.py. Round 4 fixed what some checks measured (reasons and re-scored runs: progressive.md §16.12):

- PG2 calibration: the debiased RMS calibration error (Kumar et al. 2019) over every judged claim of days ≥ 7 pooled, target ≤ 0.05. The plug-in ECE of one snapshot (~60 statements) has a floor of 0.11–0.16 for a perfectly calibrated set, so the old check could not be met by any learner; the pooled debiased error reads ~0.01 for a calibrated set (sd 0.015–0.02) and ≥ the population ECE. The plug-in ECE, its null floor and the node / group-part split are reported beside it.
- PG2 trends (recall ± 0.05, mean depth, confidence, unseen-IP mass) are read on §12's snapshot days {3, 5, 7, 10, 14, 21} outside the drift days, as defined; runs snapshot daily since round 3 and the checks had silently become day-to-day (daily recall dips are still reported). Confidence and U compare the same unchanged truth patterns on consecutive snapshot days (median paired change), so newly recovered patterns entering at a low confidence no longer read as a fall.
- Precision and calibration count each distinct claim once (a statement rendered twice is one claim).
- PG10 window: the learned edges are compared with the truth's window (central 99 %) or, when the statement states a coverage 0.5 ≤ c < 0.99, with some set of intervals of the truth law holding c, within ± 2 min; a correct 74 % window could never meet the 99 % edges.
- Reproducibility: P12 / P15 decide on a deterministic price of each engine run's counted work (lib/pcost; cost_model 'wall' optional), and the evaluation entry points pin PYTHONHASHSEED; two runs of the same seed then make the same decisions.
- Random draws (evaluator, round 4): the held-out events of each truth pattern and the precision / calibration draws of each claim come from their own stream, a function of (seed, day, purpose, item) (pmetrics._stream); before, one stream per snapshot made precision depend on the ORDER of the statements (a pure reordering moved seed 0 day 7 from 0.771 to 0.743).
- PG2 "patterns of a period reach 80 % recall by day N" is not measured (None) when no truth pattern of that period is eligible on any snapshot day (pmetrics._ttr_ok). Pack O's only weekly pattern (sales' Friday report) occurs on 2 Fridays in 21 days (day 5, a Friday, is a holiday) and was never eligible; rounds 2–4 had read that as a failure.
- What did NOT change: the recall / precision / component definitions, the held-out law, the targets (ECE ≤ 0.05 now reads the pooled debiased error) and PG10's ± 2 min. Round-3 runs re-scored with these semantics: progressive.md §16.12.3 and reports/progressive/round4_eval/rescore_pg2_pg10.json.
