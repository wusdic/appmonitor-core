# Behavior library (lib-3) — current state

Docs sync of 2026-10-01 (after the progressive-core integration). The Chinese design document
(`docs/组织业务系统画像平台设计.md`, library 3, §3.0–§3.12) is the full narrative; this page is the
short English summary with pointers. Every number below comes from `reports/progressive/` (pack O) or
`reports/eval_report.json` (packs A–E, round 4).

## What is built

Library 3 has two parts that communicate only through the MetricStore by metric / model name
(architecture.md §0a):

- **Progressive profile core (P00–P15)**, specification `progressive.md`: P00 event_builder (raw),
  P01 event_context (derived), P02 attr_registry, P03 conformity, P04 pattern_tree, P05 attr_select,
  P06 content_bounds, P07 payload_grammar, P08 binding, P09 time_window, P10 workflow, P11 who_groups,
  P12 system_profile, P13 facets, P14 views, P15 resource_governor. It learns from behaviour events
  with an open attribute map, grows a budgeted pattern tree per system (or system family) from the
  root down only where an anytime-valid e-value test, an MDL gain and a stability test pass, fits
  content / time / workflow constraints and IP → value bindings, discovers behavioural groups, scores
  every event against the most specific confirmed pattern, and renders a system view and a group view
  in Chinese and English. As-built cards and deviations: engines.md (end), `progressive.md` §16.2.
- **Statistical engine library (B01–B30)**, `backend/app/engines/behavior/`: B01–B18,
  B21 cross_system (built in round 4), B23 feedback, B24 calibration, B25 fusion, B26 risk,
  B27 incident, B28 governor, B29 explain, B30 portrait, on the upgraded raw layer (R1 full sets and
  weighted records, R2 `raw.action_token`, R3 `raw.client_stack`) and derived layer (D0 zero-filled
  window grid, D1 fresh-only instant metrics, D2 sessions). It now supplies facets and detector
  families; its decision chain B23–B30 is shared with the core (P03's `conf_*` detectors are family
  `conformity`). Bounded mode (`lib3.resource_mode = 'bounded'`, `lib/pactive.py`) limits per-IP
  models to earned IPs. B19 mixture, B20 session_profile and B22 action_embedding are specified but
  **not built** and stay behind the ablation gate.
- **Registry modes** (`build.py::build_registry(progressive=…)`): `full` (default; 44 engines, no
  P engine; default Runtime, packs A–E), `full+progressive` (60), `progressive_only` (18; scaling
  packs), `progressive_decision` (24 = P-core + B24–B29; every pack O measurement).
- **API and UI**: API v3 (`backend/app/api/routes_v3.py`, `/api/v3`: system / group / IP views,
  precision over time, pattern lattice and details, violations, facets, strategy, attributes, budget,
  group naming) and the frontend page 画像模式 (`frontend/js/progressive.js`), added in parallel with
  this docs sync (`tests/api/test_routes_v3.py`: 16 passed); `APPMON_PROGRESSIVE=decision|only|full`
  runs the backend on an organisation pack (`api/progressive_runtime.py`).
- **Not built**: `scripts/import_who_names.py`, `groups_as_classes`, `lib4_inputs`, and B23 labels
  turning into P04 allow-sets.

## How the progressive core works (one line each)

| Stage | Method | Engines |
|---|---|---|
| Events | Every request / connection / query (aggregated records expanded through `extra.ev_sample`) becomes an event with all Observation fields, `extra.meta` leaves, parsed body / query key-values and headers; secrets kept as shape; threshold sampling caps learning at e_rate events/s per tree with HT mass; metric-window events carry any fresh raw / derived metric | P00, P01 |
| Open schema | Online type inference; generalisation hierarchies (IP /32 → /24 → /16 → learned group → region → *, route, time, numeric bins, value shapes); roles split / target / invariant / shape / redundant / dropped by measured bits per event with hysteresis | P02, P05 |
| Pattern tree | Evidence units ω ≤ 1 per observed row (bursts harmonic); evidence-scaled hierarchical-Dirichlet predictives and prequential code lengths; split rule (V) averaged e-process ≥ 2^(10 + log2 C_ever) (as built: blockwise k-sample e-process), (G) selective MDL gain, (S) time-uniform empirical Bernstein, (D) ≥ 2 dates, (M) ≥ 2 groups × 5 units; route-first partition of the root; value grouping; prune / merge / EFDT revision / budget; single-IP exceptions by a leave-x-out e-value | P04 |
| Lifecycle and drift | candidate → confirmed (n_c ≥ 20 on ≥ 3 dates) → stable → evolving / stale / retired / dormant; confidence channel (H_l = 30 d, reset on accepted change) separate from shape (H_m = 7 d); ADWIN and Page–Hinkley; coordinated-change acceptance; who-set changes never by persistence alone; trust gating, damping 0.1 for p ≤ 1e-4, held events; daily reference snapshot (dual anchor) | P04 |
| Constraints | 90 % bands and exchangeability hard bounds (P(next outside) ≤ 2/(n_rng + 1)) with GPD tails; key sets and character-class grammars by anti-unification, closed value sets, `injection_shape`; approximate FDs with leave-one-out empirical-Bayes Beta lower bounds, set bindings, `cross_binding` / `concurrent_use` / `readdress_candidate`; Bayesian-Blocks activity windows with HDR p; heuristics-miner workflows and required predecessors | P06–P10 |
| Who | Weighted MinHash (ICWS) + LSH + mutual-kNN + Louvain over (tree, action) signatures; stable ids; names from `who_group_names` (configuration or inventory import; traffic gives membership, not names); prefix covers | P11 |
| Conformity | Typed p-values who (Good–Turing unseen mass of the closed level) / when / content / seq / novel with back-off and the reference snapshot; Šidák per tick → `behavior.score[conf_*]`; per-day `p_day` for discrete `pattern_violation`; per-IP intensity on guaranteed Space-Saving counts at open nodes | P03 |
| Facets and views | Runtime facet registry (10 top facets, 22 default sub-facets); statements with support, confidence, first / last seen, version; system view with per-group parts, group view with negative statements | P13, P14 |
| Adaptation and budgets | System characteristics; arms with preconditions; utility in bits per event minus λ·µs; Hedge / budgeted UCB; who-level two-part code lengths; system families sharing one tree; P15 water-filling budgets, a 7-step degradation ladder, active and earned sets | P12, P15 |

## How the statistical library works (one line each)

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

## Latest results

### Pack O — the progressive core (`reports/progressive/`)

`progressive_report.{json,html}`, per seed `runs/O_<seed>.json`; 21 days at 900 s, seeds 0–2 (seed 2 held
out), registry `progressive_decision`, strict; ≈ 261 000 events per run (582 anomalous), 27–28 min with three
runs in parallel, peak RSS 1.37–1.40 GB, no engine exception. (The report's top-level `registry` field says
the pack default `full+progressive`; each run's own `registry` field is the mode actually used.)

| Gate | Target | Seeds 0 / 1 / 2 | Pass |
|---|---|---|---|
| PG1 recall / precision @ day 14 | ≥ 0.90 / ≥ 0.90 | 0.21 / 0.18 / 0.21; 0.30 / 0.39 / 0.34 | no (workflow component 0.92 passes) |
| PG2 ECE; median confidence non-decreasing | ≤ 0.05; yes | 0.39 / 0.43 / 0.36; no (falls 0.94 → 0.12 on seed 1) | no |
| PG3 GA login = 3 IPs / finance approval = 1 IP / P11 ARI | yes / yes / ≥ 0.9 | 1 of 3 / 0 of 3 / 0.86–0.87 | no |
| PG4 memory slope vs IPs / attributes / systems | ≤ 0.15 / 0.2 / 0.3 | 0.085 / 0.23 / 0.21 | IPs, systems yes; attributes no |
| PG4 scoring / learning p95 per event | ≤ 100 / 250 µs | 0.2–2.1 / 1.6–5.0 ms | no |
| PG5 D2 rebinding (mike → mike.w) | ≤ 5 events, 2 d | pass on all seeds | yes; D1, D3, D4, D5 and non-adoption fail |
| PG6 anomalies A1–A10 (violation + incident) | ≥ 0.95 | 7 / 6 / 6 of 10 | no |
| PG6 FAR incidents ≥ LOW / ≥ MEDIUM per entity-day | ≤ 0.10 / ≤ 0.02 | 0.027–0.032 / 0.025–0.029 | LOW yes, MEDIUM no |
| PG7 registered first tick / role ≤ 24 h / type correct | 1 / 1 / ≥ 0.95 | 1 / 1 / 0.67 | partly |
| PG8 strategy arms in truth | ≥ 0.95 | 0.33 / 0.67 / 0.50 | no |
| PG10 view checks | all | 0 of 4 | no |
| PG9, PG11, O-red, O60, five seeds | – | not run | – |

The requirement's example as learned (seed 0; the design doc §3.8.5 quotes the statements): the approval
and 17:00-report activities with their actors, windows, sizes and predecessor page are recovered on all
seeds; the finance approval statement is "192.168.2.10 only" on days 11 and 14 and widens to /24 on day 21
after A8's one-off source; 综合部's group view says "never writes in finance" on day 14 and correctly no
longer on day 21, after A1. The 综合部 login node holds its three IPs on seed 0 only, from day 18; one of
three bindings is right (A2's borrowed `rose` is adopted for 192.168.1.21). Anomalies detected on all
seeds: A1, A3, A4, A6, A8, A10; A2 on seed 0 only; A5, A7, A9 not.

### Packs A–E — the statistical library (`reports/eval_report.json`, round 4)

Strict, canonical; packs A–E and mini seeds 0 and 1, plus smoke (commit 1473bae). Gate 15 (robustness)
passes, 12 and 13 are n/a, the other 12 gates fail.

| Gate | Round 4 | Target |
|---|---|---|
| 1 threat recall in deadline (loud / subtle) | 0.71 (0.45 / 0.84) | 0.95 (1.0 / 0.9) |
| 3 FAR ≥ LOW / ≥ MEDIUM per entity-day | 0.156 / 0.042 (both pass) | 0.2 / 0.05 |
| 3 HIGH+ / CRITICAL on control entities, all runs | 9 / 2 | ≤ 1 / 0 |
| 4 legit runs within allowed severity | 0.33 | 0.95 |
| 5 notifications per TP incident | 3.03 | 3 |
| 7 worst median KS D; single-tick exceedance cc 900 / cc 60 | 0.19; 5.3× / 3.0× | 0.05; [0.5, 2]× |
| 7 evidence-CUSUM alarms per entity-day | 0.051 | 0.045 |
| 9 role ARI | 0.75 | 0.9 |
| 10 hit@3 / counterfactual validity | 0.71 / 0.98 (validity passes) | 0.8 / 0.9 |
| 14 max pack-seed wall / live p95 at 35 entities | 2018 s / 1246 ms (4 processes in parallel) | 360 s / 80 ms |

The default `full` registry gives the same results as round 4 after the progressive integration (smoke and
mini seed 0 re-checked after the last engine changes: 0 differences except timings).

Tests: full suite 2 769 passed, 4 skipped, 983 s (the integrator's run after all changes of the progressive
integration). The APPMON_SLOW Part B test `test_part_b_live_60_equals_900` is left failing on purpose
(60-s t-stream calibration).

## Top open issues (design doc §3.11)

1. P04 splits the OA login node first by TCP-window class (client-stack group) and only later by
   department, so 综合部's node and its bindings arrive late (PG1 who, PG3, PG10, A2, PG5).
2. Damped sources still count in P04's who mass and P14's rendering (A9 and A8 appear in the finance
   approval statement; the IP level does not close).
3. Stated confidence neither rises with time nor is calibrated (ECE 0.36–0.43): "longer is more precise"
   is not demonstrated for the stated confidence.
4. ≥ MEDIUM FAR 0.025–0.029: a novel-action feedback loop on SALES' weekly report (incident → quarantine
   → P10 does not learn the action) needs labels → P04 allow-sets (not built) or group-level adoption.
5. Undetected A5, A7, A9 (no incident below MEDIUM), A2 on seeds 1–2; drift D1, D3, D4, D5 fail.
6. P12 strategy choice 0.33–0.67; per-event cost 10–20× the target; attribute memory slope 0.23; the
   decision chain B24–B28 still keeps per-(system, IP) state and costs more than the core on pack O.
7. Not run: five seeds, O-red, O-real (PG11), O60, PG9 (packs A–E with the core on and in bounded mode),
   300-system and 7-day scaling points, pack O with the full B library (> 7 GB per run).
8. Statistical library (round-3 list, design doc §3.11.2): 60-s t-stream calibration, gate 14 CPU, the
   identification / class / explanation sub-gates.

## Document map

- `progressive.md` — the progressive core: requirement traceability (§3), data model (§5), algorithms with
  formulas (§6), budgets (§7), engine cards (§8), integration with B01–B30 (§9), bounded mode (§10),
  generator (§11), gates PG1–PG11 (§12), decisions and risks (§14), integration changes and measured
  results (§16).
- `architecture.md` layered design (§0a: the two parts of library 3); `engines.md` per-engine specs with
  as-built and canonical-mode notes, P00–P15 cards at the end; `cadence.md` the v2.1 dual-grain design
  with implementation notes (§17, §17.1).
- `contract.md` store contract (N: progressive additions); `helpers_api.md` shared maths API.
- `eval.md` gates of packs A–E; `generator.md` personas, packs and scenarios of packs A–E (pack O:
  `progressive.md` §11).
- `integration.md` registry, integration fixes, cost, evaluation rounds (§8–§12) and the progressive-core
  integration (§13).
- `decisions.md` dropped alternatives and risks; `api_ui.md` API / UI spec.

## History

This file previously held the design-review verdict on the v1 library (stale
carry-forward, re-stamped derived inputs, ungated learning, per-run
IsolationForest, KMeans archetypes, cosine drift and separability, truncated
categorical sets, tick-based thresholds, linear store scans) and the list of
revisions the v2 design made in response. Every item of that list is
implemented; the v1 engines are deleted (integration.md §2) and the
alternatives are recorded in `decisions.md`.
