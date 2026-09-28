# Behavior library (lib-3) — current state

Docs sync after evaluation round 3 (2026-09-28). The Chinese design document
(`docs/组织业务系统画像平台设计.md`, library 3) is the full narrative; this page
is the short English summary with pointers.

## What is built

- **26 registered behaviour engines** (`backend/app/engines/behavior/`), run in
  this order by `backend/app/pipeline/build.py::build_registry()`:
  B01 feature_vector, B02 peer_group, B03 baseline, B04 likelihood,
  B05 common_mode, B06 multivariate, B07 rhythm, B08 novelty,
  B09 client_identity, B10 sequence, B11 timing, B12 beacon, B13 budget,
  B14 changepoint, B15 identity_model, B16 attribution, B17 entity_link,
  B18 class_monitor, B23 feedback, B24 calibration, B25 fusion, B26 risk,
  B27 incident, B28 governor, B29 explain, B30 portrait.
- Library-3 inputs from the upgraded raw layer (R1 full sets and weighted
  records, R2 `raw.action_token`, R3 `raw.client_stack`) and the derived layer
  (D0 zero-filled window grid, D1 fresh-only instant metrics, D2 sessions).
- P2 engines B19 mixture, B20 session_profile, B21 cross_system and
  B22 action_embedding are specified (engines.md) but **not built** and not
  registered; they stay behind the ablation gate.
- Engines communicate only through the MetricStore by metric / model name;
  shared maths is in pure modules under `engines/behavior/lib/`.

## How it works (one line each)

| Stage | Method | Engines |
|---|---|---|
| Representation | 52 features in 9 groups with explicit exposure, time context (tz, holidays, make-up days), 80-dim token sketch; absence is data | B01 |
| Cadence invariance (spec v2.1, default `grain_mode='canonical'`) | Features on canonical trailing wall-clock grains H = 3600 s and Q = 900 s, built from additive parts, mergeable set sketches and count maps; scored rows only on epoch-aligned decision ticks; midpoint time context | B01, `lib/grains.py`, cadence.md |
| Baselines | Conjugate hierarchical Bayes (entity → role class → system → org → hyperprior, EB pseudo-counts), bin48/bin168 buckets, current (0.1σ15/day) and reference (0.03σ15/day, 24 h delay, golden) anchors, trust-gated delayed reversible commits; Q predictive = native Q anchor + κ_T = 16 pseudo-rows of the H predictive transferred with v = 1 + ((m−1)/m)ω | B03 |
| Exact predictives | NB / Beta-Binomial / Student-t two-sided mid-p against both anchors, p = min(1, 2 min(p_cur, p_ref)); wHMP within and across dependence groups | B04 |
| Detectors | LOO common mode; OAS / C-step T² and SPE with RBC; 15-min slot rhythm (Bernoulli CUSUM, silence, automation index); hierarchical Dirichlet + Good–Turing novelty; client-stack impersonation; PPM-C grammar; timing; Gamma renewal LRT beacon with Monte-Carlo null; POT/GPD budgets; CUSUM / MCUSUM / BOCPD / creep on reference residuals | B05–B14 |
| Identity | Absolute 4-active-H-row windows, OAS-WCCN-LDA with blocked CV (EER_hard, separability), calibrated capped modality LLRs, open-set attribution with other-identity and unknown CUSUMs, Fellegi–Sunter linking and shared-IP tests | B15–B17 |
| Classes as entities | Two-level HDBSCAN hierarchy (role, individual sub-class) plus static CIDR and pool classes; class aggregate baselines and detectors (class_int, class_shape, class_rhythm, class_novel, class_coherence) | B02, B18 |
| Calibration | Randomised Mondrian conformal p per (key, detector, stratum), GPD tail, small-sample prior with randomised pm atoms and an own-history floor, KS / exceedance health | B24 |
| Fusion | wHMP per family, per-entity meta-calibration with a winsorised tail, single-tick e_day = q_all·n_τ/β_τ over tick types (β = 0.5 / 0.25 / 0.25), one evidence CUSUM per stream (S_t, S_h; ARL 66 d each), accumulator paths, corroborated severities, axis reading rules | B25 |
| Decision and governance | Cadence-invariant decaying risk with fixed L_ref = 60; incidents that close on regime / accumulators / evidence, never on risk; regime machine with legit-vs-attack log-odds, rollback / release / rebase, corroborated REJECT, no permanent lockout; analyst feedback (precision priors, stacking weights, pattern policies, alert budget, label queue) | B23, B26–B28 |
| Explanation and portraits | Natural-unit attribution, faithful counterfactual by deterministic replay of stateful detectors; versioned per-IP and per-class portraits with per-grain bands and diffs | B29, B30 |

## Latest evaluation (reports/eval_report.json, round 3 final)

Strict, canonical; packs A and B seeds 0–1, C / D / E seed 0 (the 4 cores were
shared with a leftover batch, so gate-14 timings are contended). Of the 15
gates only **15 (robustness)** passes; 13 is n/a in the final report (ablation
from an older tree); the other 13 fail.

| Gate | Final | Target |
|---|---|---|
| 1 threat recall in deadline (loud / subtle) | 0.58 (0.36 / 0.68) | 0.95 (1.0 / 0.9) |
| 3 FAR ≥ LOW / ≥ MEDIUM per entity-day | 0.237 / 0.152 | 0.2 / 0.05 |
| 3 HIGH+ / CRITICAL on control entities, all runs | 60 / 19 | ≤ 1 / 0 |
| 4 legit runs within allowed severity | 0.34 | 0.95 |
| 5 notifications per TP incident | 7.05 | 3 |
| 7 worst median KS D; single-tick exceedance cc 900 / cc 60 | 0.41; 64× / 2 818× | 0.05; [0.5, 2]× |
| 7 evidence-CUSUM alarms per entity-day | 1.20 | 0.045 |
| 8 twins confusable / Spearman / T19 chain | 0.14 / −0.09 / 0 | 1 / 0.8 / 0.9 |
| 9 role ARI / refit ARI | 0.60 / 0.74 | 0.9 / 0.95 |
| 10 hit@3 / counterfactual validity | 0.25 / 0.35 | 0.8 / 0.9 |
| 12 feedback cut of control incidents | −0.08 | 0.5 |
| 14 pack-seed wall / live p95 at 35 entities | 2874 s / 1403 ms (contended); 740 s / 909 ms uncontended pack A (§9) | 360 s / 80 ms |

Tests: full suite 2185 passed, 4 skipped. The APPMON_SLOW Part B test
`test_part_b_live_60_equals_900` fails (60-s t-stream calibration) and is left
failing on purpose.

## Top open issues (integration.md §10.7, §11)

1. Recurring lib-4 HIGH matches in a clean warm-up (sanctioned backups, NAT):
   no habituation, zero training trust, B13 falls back to peers; L15 is
   CRITICAL in every run. Needs a design decision.
2. 60-s t-stream calibration after a 900 → 60 s switch (empty cc = 60 strata,
   accumulators in p_all, evidence cap).
3. lib-4 ratio / entropy clauses still per tick at 60 s.
4. Eval metric: an attack that escalates an already-open LOW FP incident is
   not counted as a detection.
5. Detector calibration at 900 s (identity, timing, spe, marg_shape_q,
   budget_vol 12–21× nominal at p < 1e-3); idle API clients at risk 20–30.
6. Loud TTD ≤ 2 ticks at 900 s is out of reach under the conservative H → Q
   transfer.
7. Gate 12 (labels increase control incidents), gate 14 CPU and 60-s memory,
   and the identification / class / explanation sub-gates listed above.

## Document map

- `architecture.md` layered design and principles; `engines.md` per-engine
  specs with as-built and canonical-mode notes; `cadence.md` the v2.1
  dual-grain design with implementation notes (§17, §17.1).
- `contract.md` store contract; `helpers_api.md` shared maths API.
- `eval.md` gates; `generator.md` personas, packs and scenarios.
- `integration.md` registry, integration fixes, cost, evaluation rounds
  (§8–§10) and the round-2 summary (§11).
- `decisions.md` dropped alternatives and risks; `api_ui.md` API / UI spec.

## History

This file previously held the design-review verdict on the v1 library (stale
carry-forward, re-stamped derived inputs, ungated learning, per-run
IsolationForest, KMeans archetypes, cosine drift and separability, truncated
categorical sets, tick-based thresholds, linear store scans) and the list of
revisions the v2 design made in response. Every item of that list is
implemented; the v1 engines are deleted (integration.md §2) and the
alternatives are recorded in `decisions.md`.
