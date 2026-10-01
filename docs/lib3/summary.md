# Behavior library (lib-3) — current state

Docs sync of 2026-10-01, updated after round 2 of the progressive core (final code, 5 seeds). The Chinese design document
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
  models to earned IPs and (round 2) the decision chain's per-IP state to earned and active IPs, released by
  P15 after 7 idle days. B19 mixture, B20 session_profile and B22 action_embedding are specified but
  **not built** and stay behind the ablation gate.
- **Registry modes** (`build.py::build_registry(progressive=…)`): `full` (default; 44 engines, no
  P engine; default Runtime, packs A–E), `full+progressive` (60), `progressive_only` (18; scaling
  packs), `progressive_decision` (24 = P-core + B24–B29; every pack O measurement and, since round 2, the default of
  the organisation packs). In bounded resource mode `full` also registers P15 (45).
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
| Pattern tree | Evidence units ω ≤ 1 per observed row (bursts harmonic); evidence-scaled hierarchical-Dirichlet predictives and prequential code lengths; split rule (V) averaged e-process ≥ 2^(10 + log2 C_ever) (as built: blockwise k-sample e-process), (G) selective MDL gain, (S) time-uniform empirical Bernstein, (D) ≥ 2 dates, (M) ≥ 2 groups × 5 units; route-first partition of the root; value grouping; prune / merge / EFDT revision / budget; single-IP exceptions by a leave-x-out e-value. Round 2: a split is paid only by the behaviour it explains (never by `@who` or source properties such as client stack / TCP window), who levels first and offered as a ladder down to /24, children seeded from the parent's per-value counts, constancy judged over two days | P04 |
| Lifecycle and drift | candidate → confirmed (n_c ≥ 20 on ≥ 3 dates) → stable → evolving / stale / retired / dormant; confidence channel (H_l = 30 d, reset on accepted change) separate from shape (H_m = 7 d); ADWIN and Page–Hinkley; coordinated-change acceptance; who-set changes never by persistence alone; trust gating, damping 0.1 for p ≤ 1e-4, held events; daily reference snapshot (dual anchor). Round 2: stated confidence = (passes + 1) / (tests + 2) of held-out tests of everything the statement states (≥ 100 events or a week); sources P03 damped as foreign are suspects and never enter a node's who | P04 |
| Constraints | 90 % bands and exchangeability hard bounds (P(next outside) ≤ 2/(n_rng + 1)) with GPD tails; key sets and character-class grammars by anti-unification, closed value sets, `injection_shape`; approximate FDs with leave-one-out empirical-Bayes Beta lower bounds, set bindings, `cross_binding` / `concurrent_use` / `readdress_candidate`; Bayesian-Blocks activity windows with HDR p; heuristics-miner workflows and required predecessors | P06–P10 |
| Who | Weighted MinHash (ICWS) + LSH + mutual-kNN + Louvain over (tree, action) signatures; stable ids; names from `who_group_names` matched on addresses (configuration or inventory import; traffic gives membership, not names); a learned group inside a configured department is one of its roles; prefix covers | P11 |
| Conformity | Typed p-values who (Good–Turing unseen mass of the closed level) / when / content / seq / novel with back-off and the reference snapshot; Šidák per tick → `behavior.score[conf_*]`; per-day `p_day` for discrete `pattern_violation`; per-IP intensity on guaranteed Space-Saving counts at open nodes | P03 |
| Facets and views | Runtime facet registry (10 top facets, 22 default sub-facets); statements with support, confidence, first / last seen, version; system view with per-group / per-department parts, group view and department view (`class:grp:dept:<name>`, composing the department's roles) with negative statements | P13, P14 |
| Adaptation and budgets | System characteristics; arms with preconditions; utility in bits per event minus λ·µs; Hedge (one round per completed day) / budgeted UCB; the who arm by held-out behaviour gain (round 2; address code only as a tie-break); system families sharing one tree; P15 water-filling budgets, a 7-step degradation ladder, active and earned sets | P12, P15 |

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

### Pack O — the progressive core (`reports/progressive/`, round 2)

`progressive_report.{json,html}`, per seed `runs/O_<seed>.json` (round-1 runs kept in `round1/runs/`); final
code of 2026-10-01 (commit 53cd151 + the owners' round-2 changes + the evaluator's E1–E11; deviations in
`progressive.md` §16.9 and §16.10.1); 21 days at 900 s, seeds 0–4 (0–1 development, 2–4 held out),
`progressive_decision` (now the organisation packs' default), strict; ≈ 261 000 events per run (582
anomalous), 29–31 min with three runs in parallel, peak RSS 1.26–1.30 GB. Medians [min–max]; round 1
(seeds 0–2) in the last column.

| Gate | Target | Round 2 (5 seeds) | Round 1 |
|---|---|---|---|
| PG1 recall / precision @ day 14 | ≥ 0.90 / ≥ 0.90 | 0.58 [0.53–0.58] / 0.53 [0.52–0.57] | 0.21 / 0.34 |
| PG1 components who / when / content / bindings / workflow | ≥ 0.85 each | 0.87 / 0.84 / 0.61 / 0.67 / 0.92 | 0.40 / 0.66 / 0.53 / 0.67 / 0.92 |
| PG1 recall over days 5 / 10 / 14 / 21 | rising | 0.22 / 0.55 / 0.58 / 0.60 | 0.16 / 0.23 / 0.21 / 0.29 |
| PG2 ECE @ 14 / @ 21; median confidence non-decreasing | ≤ 0.05; yes | 0.30 / 0.36; no (day 7 0.43 → day 21 0.35) | 0.39 / 0.33; no |
| PG3 GA login = 3 IPs / finance approval = 1 IP / portal at prefix / DEV as its pool / P11 ARI | yes ×4 / ≥ 0.9 | 4/5 / 5/5 / 5/5 / 5/5 / 0.974 | 1/3 / 0/3 / 0/3 / 0/3 / 0.86 |
| PG4 memory slope vs IPs / attributes (7-day points) | ≤ 0.15 / 0.2 | 0.036 / 0.147 | 0.085 / 0.23 (3-day) |
| PG4 CPU / event vs attributes; scoring / learning p95 | ≤ 0.2; ≤ 100 / 250 µs | 0.236; 0.42–0.70 / 1.8–2.7 ms per scored / learned event | 0.12; 0.2–2.1 / 1.6–5.0 ms per generated event |
| PG5 D1 / D2 / .21 still `jack` / D3 / D4 / D5 | yes | 3/5 / 5/5 / 5/5 / 0/5 / 0/5 / 0/5 | 0/3 / 3/3 / 0/3 / 0/3 / 0/3 / 1/3 |
| PG6 anomalies A1–A10 | ≥ 0.95 | 48/50 (A4 seed 3, A2 seed 4 missed) | 19/30 |
| PG6 FAR incidents ≥ LOW / ≥ MEDIUM per entity-day | ≤ 0.10 / ≤ 0.02 | 0.005 / 0.004 | 0.031 / 0.029 |
| PG6 B29 top reason = violated constraint | ≥ 0.9 | 0 | 0 |
| PG7 registered / role ≤ 24 h / type correct | 1 / 1 / ≥ 0.95 | 1 / 1 / 1.0 (pass) | 1 / 1 / 0.67 |
| PG8 strategy arms in truth; switches after day 7 | ≥ 0.95; ≤ 2 | 0.83 [0.67–0.83]; ≤ 2 | 0.50; 3 |
| PG9 bounded vs full (packs A, E seed 0) | no loss | detections unchanged; latency 63–87 % | not run |
| PG10 finance single IP / GA negative / OA statements day 11, 21 | all | 5/5 / 5/5 / 0/5 | 0/3 / 0/3 / 0/3 |
| PG11 (O-real) | – | not run | not run |

The requirement's example as learned (design doc §3.8.0a quotes P14's statements verbatim, both views; English
in `progressive.md` §16.10.6): on 4 of 5 seeds the tree isolates a 综合部 login node naming exactly
192.168.1.21, .23 and 10.168.7.121, with `username=` required, grammar `[a-z]{4}(\.[a-z])?` (seed 0 also the
closed set {jack, mike, mike.w, rose}) and the three bindings (192.168.1.21 still `jack` on day 21 although A2
borrowed `rose` on days 17–21). Before D1, seeds 1–4 state "工作日 09:00–09:21 / 09:00–09:20 / 09:03–09:20 / …,
综合部（3 个 IP）访问 POST /login" on day 11. On all 5 seeds: .21's approvals, the 17:00 report by .23 / .121
with its form page, "192.168.2.10 only" for finance approvals, and the department view "综合部 在 finance 中从未
执行写操作 …；192.168.1.23 的尝试被判定为越权（未学习）". Not recovered: "all logins in 0.5–3 KB" (1/5; the
department node starts with an empty numeric summary when it is split off on day 10–15).

Variants: O-red seed 0 (never tuned on) recall @ 14 0.41, anomalies 7/10, false splits 0; O60 seed 0 day-8
recall 0.26 vs pack O's 0.21 (difference 0.053 against a 0.05 tolerance); O-real not run.

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

The default `full` registry gives the same results as round 4 after the progressive integration and round 2
(packs A and E seed 0 re-run on the final code: identical to `reports/round4/runs` key for key, apart from timings).

Tests: full suite 2 881 passed, 4 skipped, 21 min (the evaluator's run on the final round-2 tree). The APPMON_SLOW Part B test `test_part_b_live_60_equals_900` is left failing on purpose
(60-s t-stream calibration).

## Top open issues (design doc §3.11.1, `progressive.md` §16.10.8–§16.10.9)

1. Pattern recall 0.58 (target 0.90): mail departments differ only in their windows (nodes /16 or mixed);
   closed value sets capped by `TEXT_VALUES_K` = 16 and late for three-user finance; required form keys not
   requested where `body.keys` is not a P05 target; daily two-person actions confirm after day 14.
2. Statement confidence (held-out test-pass frequency, M46) is neither rising with time nor calibrated
   (ECE 0.30): it under-states holding (0.39 vs 0.52 on seed 0); most failing checks are `when`, where the
   truth's step windows are the whole activity's.
3. A split child starts with empty numeric summaries ("all logins 0.5–3 KB" 1/5); on seed 4 the 综合部
   login node is split off late (2 IPs, no binding, A2 missed).
4. P03's time p-value floor α/(N + α) on young nodes (A4 missed on seed 3); D3 route renames never adopted
   (a successor rule needs a decision); D4 / D5 LOW incidents.
5. Suspect sources are never cleared while they send rows (M29, by design against slow poisoning) and B28
   keeps trust 0 while any incident is open — both need lead decisions.
6. P12 0.83: finance's username pair is screened late (P08) and its who switch is late; DEV pool not one
   P11 group; per-event cost of P03 / P04 2–10× the target; CPU slope vs attributes 0.236.
7. Not run: O-real (PG11), PG9 on packs B–D and seeds 1–4, the 20 000-IP and 300-system points, pack O
   with the full B library (> 7 GB per run).
8. Statistical library (round-3 list, design doc §3.11.2): 60-s t-stream calibration, gate 14 CPU, the
   identification / class / explanation sub-gates.

## Document map

- `progressive.md` — the progressive core: requirement traceability (§3), data model (§5), algorithms with
  formulas (§6), budgets (§7), engine cards (§8), integration with B01–B30 (§9), bounded mode (§10),
  generator (§11), gates PG1–PG11 (§12), decisions and risks (§14), integration changes and measured
  results (§16; round 2 in §16.9–§16.10, status per requirement sentence in §3.1).
- `architecture.md` layered design (§0a: the two parts of library 3); `engines.md` per-engine specs with
  as-built and canonical-mode notes, P00–P15 cards at the end; `cadence.md` the v2.1 dual-grain design
  with implementation notes (§17, §17.1).
- `contract.md` store contract (N: progressive additions); `helpers_api.md` shared maths API.
- `eval.md` gates of packs A–E; `generator.md` personas, packs and scenarios of packs A–E (pack O:
  `progressive.md` §11).
- `integration.md` registry, integration fixes, cost, evaluation rounds (§8–§12), the progressive-core
  integration (§13) and its round 2 (§14).
- `decisions.md` dropped alternatives and risks; `api_ui.md` API / UI spec.

## History

This file previously held the design-review verdict on the v1 library (stale
carry-forward, re-stamped derived inputs, ungated learning, per-run
IsolationForest, KMeans archetypes, cosine drift and separability, truncated
categorical sets, tick-based thresholds, linear store scans) and the list of
revisions the v2 design made in response. Every item of that list is
implemented; the v1 engines are deleted (integration.md §2) and the
alternatives are recorded in `decisions.md`.
