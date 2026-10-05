# Behavior library (lib-3) — current state

Docs sync of 2026-10-05, updated after round 4 of the progressive core (final code, 5 seeds). The Chinese design document
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
| Pattern tree | Evidence units ω ≤ 1 per observed row (bursts harmonic); evidence-scaled hierarchical-Dirichlet predictives and prequential code lengths; split rule (V) averaged e-process ≥ 2^(10 + log2 C_ever) (as built: blockwise k-sample e-process), (G) selective MDL gain, (S) time-uniform empirical Bernstein, (D) ≥ 2 dates, (M) ≥ 2 groups × 5 units; route-first partition of the root; value grouping; prune / merge / EFDT revision / budget; single-IP exceptions by a leave-x-out e-value. Round 2: a split is paid only by the behaviour it explains (never by `@who` or source properties such as client stack / TCP window), who levels first and offered as a ladder down to /24, children seeded from the parent's per-value counts, constancy judged over two days. Round 3: split children are born with their own history (≤ 128 replayed rows per learning leaf plus exact per-value extremes), new route nodes get the ≤ 8 rows they waited for, a split never names "not grouped yet" (`grp:∅`) as a child Round 4: every leaf keeps exact per-source and per-/24 extremes from its birth (T-1). | P04 |
| Lifecycle and drift | candidate → confirmed (n_c ≥ 20 on ≥ 3 dates) → stable → evolving / stale / retired / dormant; confidence channel (H_l = 30 d, reset on accepted change) separate from shape (H_m = 7 d); ADWIN and Page–Hinkley; coordinated-change acceptance; who-set changes never by persistence alone; trust gating, damping 0.1 for p ≤ 1e-4, held events; daily reference snapshot (dual anchor). Round 2: stated confidence = (passes + 1) / (tests + 2) of held-out tests of everything the statement states (≥ 100 events or a week); sources P03 damped as foreign are suspects and never enter a node's who. Round 3: confirmation on 20 undecayed observations; suspicion is a sequential evidence test cleared only by colleagues' use; the confidence prior is fitted per statement kind (empirical Bayes); B28 keeps learning a source whose open incident concerns one learned pattern only Round 4: a held-out test is ≥ 50 independent source-days or a week, a constraint failing below nominal − max(3 σ binomial, 2 σ cluster-robust); the record forgets per test; a statement takes its kind's current prior at each of its own tests; who is stated at the share its listed sources hold (T-2 – T-6, EV-4, EV-8). | P04, B28 |
| Constraints | 90 % bands and exchangeability hard bounds (P(next outside) ≤ 2/(n_rng + 1)) with GPD tails; key sets and character-class grammars by anti-unification, closed value sets, `injection_shape`; approximate FDs with leave-one-out empirical-Bayes Beta lower bounds, set bindings, `cross_binding` / `concurrent_use` / `readdress_candidate`; Bayesian-Blocks activity windows with HDR p; heuristics-miner workflows and required predecessors. Round 3: exact-value capacity grows for finite populations (≤ 64), bindings pool a source's clean days across the tree, route renames are adopted, single-source windows need 5 dates Round 4: a set closes when its Good–Turing unseen mass is below its rarest member's share; windows keep only segments still supported, cross-validate their coverage, do not fit a change younger than 3 dates (marked drift) and state the smallest day-type coverage; bands fit on clean rows; the violation ledger feeds key presence and windows (C-1 – C-6). | P06–P10 |
| Who | Weighted MinHash (ICWS) + LSH + mutual-kNN + Louvain over (tree, action) signatures; stable ids; names from `who_group_names` matched on addresses (configuration or inventory import; traffic gives membership, not names); a learned group inside a configured department is one of its roles; prefix covers. Round 3: address pools (configured or learned) are one group covering the prefix; one-address role groups; ids matched on addresses with a lineage Round 4: checked, unchanged (stable from day 5). | P11 |
| Conformity | Typed p-values who (Good–Turing unseen mass of the closed level) / when / content / seq / novel with back-off and the reference snapshot; Šidák per tick → `behavior.score[conf_*]`; per-day `p_day` for discrete `pattern_violation`; per-IP intensity on guaranteed Space-Saving counts at open nodes. Round 3: a young node's `when` p backs off to its parent; cross-bindings checked up the ancestors Round 4: a source's renamed page is a rename, not a new action; the GPD tail is predictive (parameter uncertainty); improbable transitions are re-scored at route level (G-1 – G-3); B29 explains a pattern violation by the violated constraint (EV-6). | P03 |
| Facets and views | Runtime facet registry (10 top facets, 22 default sub-facets); statements with support, confidence, first / last seen, version; system view with per-group / per-department parts, group view and department view (`class:grp:dept:<name>`, composing the department's roles) with negative statements Round 4: duplicates folded, variant conditions stated, configured pools resolved, statements ordered and ending with confidence and support (G-4 – G-6); drifting windows stated as evolving; group parts with their own windows, bands, closed sets and confidence (EV-2, EV-3, EV-5, EV-7, T-7). | P13, P14 |
| Adaptation and budgets | System characteristics; arms with preconditions; utility in bits per event minus λ·µs; Hedge (one round per completed day) / budgeted UCB; the who arm by held-out behaviour gain (round 2; address code only as a tie-break); system families sharing one tree; P15 water-filling budgets, a 7-step degradation ladder, active and earned sets Round 4: arm costs and budget shares are a deterministic price of counted work (`lib/pcost.py`), families join on the days both systems were observed (E4-1, E4-9). | P12, P15 |

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

### Pack O — the progressive core (`reports/progressive/`, round 4)

`progressive_report.{json,html}`, per seed `runs/O_<seed>.json` (variants `runs/{O-red,O60,O-real}_0.json`); final
code of 2026-10-05 (commit 303dac6 / cbe4f95 checkpoints + the four owners' round-4 changes T-1 – T-7, C-1 – C-6,
G-1 – G-6, E4-1 – E4-11 + the evaluator's EV-1 – EV-11; `progressive.md` §16.12, each with a regression test that fails
without it); 21 days at 900 s, seeds 0–4 (0–1 development, 2 confirmation, 4 untouched before the final runs),
`progressive_decision`, `PYTHONHASHSEED=0`, resumable runs, 33–35 min each, peak RSS 1.97–2.03 GB (O60 and O-real ran
on the code before EV-10 / EV-11, O-red on the code before EV-7 / EV-8 — rendering-only differences, §16.12.7). Round 3 is shown
**re-scored** by this round's scorer (same generator, truth and held-out law; this round changed how PG2 reads
calibration and trends and how PG10 reads a window, §16.12.1) and, where it differs, as published.

| Gate | Target | Round 4 (5 seeds) | Round 3 re-scored | Round 3 as published |
|---|---|---|---|---|
| PG1 recall / precision @ day 14 | ≥ 0.90 / ≥ 0.90 | 0.84 [0.82–0.92] / 0.86 [0.84–0.87] | 0.79 / 0.69 | 0.79 / 0.69 |
| PG1 components who / when / content / bindings / workflow | ≥ 0.85 each | 0.92 / 0.95 / 0.92 / 1.00 / 1.00 | 0.92 / 0.92 / 0.84 / 1.00 / 1.00 | same |
| PG1 recall / precision @ day 21 | – | 0.85 / 0.92 | 0.71 / 0.81 | 0.71 / 0.79 |
| PG1 recall over days 5 / 10 / 14 / 21 | rising | 0.39 / 0.78 / 0.84 / 0.85 | 0.35 / 0.73 / 0.79 / 0.71 | same |
| PG2 calibration error (debiased, days ≥ 7 pooled); median confidence day 7 → 21 | ≤ 0.05; rising | 0.157 (stated 0.78, held 0.87); 0.65 → 0.91 | 0.324 (0.51 / 0.76); 0.55 → 0.62 | ECE@14 0.23 |
| PG3 GA login = 3 IPs / finance approval = 1 IP / portal at prefix / DEV pool one group / ARI | yes / ≥ 0.9 | 5/5 / 5/5 / 5/5 / 5/5 / 0.974 (pass) | same | same |
| PG4 memory vs IPs / attributes / systems; per-event p95 | ≤ 0.15 / 0.2 / 0.3; ≤ 100 / 250 µs | 0.040 / 0.172 / **0.267** (pass); 6.8 / 6.4 ms | – | 0.040 / 0.172 / 0.45; 6.4 / 5.6 ms |
| PG5 D1 / D2 / D3 / D4 / D5 / .21 still `jack` | yes | 3/5 / 5/5 / **5/5** / 3/5 / **5/5** / 5/5 | 3/5 / 5/5 / 0/5 / 0/5 / 1/5 / 5/5 | same |
| PG6 anomalies A1–A10; FAR ≥ LOW / ≥ MEDIUM; B29 top reason | ≥ 0.95; ≤ 0.10 / ≤ 0.02; ≥ 0.9 | 50/50; 0.0027 / 0.0014; 0.92 (gate passes) | 50/50; 0.0048 / 0.0034; 0 | 50/50; 0.005 / 0.003; 0 |
| PG7 registered / role ≤ 24 h / type correct | 1 / 1 / ≥ 0.95 | 1 / 1 / 1.0 (pass) | same | same |
| PG8 arms in truth; switches after day 7; who within 5 % | ≥ 0.95; ≤ 2; yes | 0.83; ≤ 1; 2/5 (decisions now reproducible) | 0.83; ≤ 1; 2/5 | same |
| PG10 finance single IP / GA negative / OA statement day 11 / day 21 | all | 5/5 / 5/5 / 1/5 / 3/5 | 5/5 / 5/5 / 1/5 / 3/5 | 5/5 / 5/5 / 1/5 / 2/5 |
| PG11 (O-real, 1 seed) | – | 4 of 12 R items (R7, R8, R10, R11) on the final code | – | 3 of 12 on an intermediate snapshot |

The requirement's example as learned (design doc §3.8.0a quotes P14's statements verbatim, both views; also
`progressive.md` §16.12.13): 49 of 50 checklist clauses pass (round 3: 47). On all 5 seeds the tree isolates a 综合部
login node naming exactly 192.168.1.21, .23 and 10.168.7.121 with `username=` required, grammar `[a-z]{4}(\.[a-z])?`,
the closed set {jack, mike, mike.w, rose} and the three bindings; seed 0, day 21: "工作日 08:32–08:51（覆盖 85 %，6 个工作日），
综合部（10.168.7.121、192.168.1.21、192.168.1.23）访问 POST /login（登录）：…提交数据量 90 % 在 1–2 KB，全部在 1–2.8 KB…。置信 0.90 · 依据 47 次".
.21's approvals are stated under their renamed pages ("（原 POST /approval/{num}/approve，页面已更名）"), the 17:00 report
with its form page, "财务部（192.168.2.10）" alone for finance approvals, and the department view says "综合部 在 finance 中
从未执行写操作 …；192.168.1.23 的尝试被判定为越权（未学习）". The one miss: seed 3's 90 % band (fitted 956–2 859 B on
n_eff 25.6, shown 1–3 KB; sampling).

Variants: O-red seed 0 (never tuned on; statements rendered by the views before EV-7 / EV-8) recall @ 14 0.57, precision 0.78, anomalies 9/10, false splits 0; O60 seed 0
day-8 recall 0.50 vs pack O's 0.55 (within 0.05); O-real once on the final code (recall @ 14 0.45, precision 0.27,
4/10 anomalies, statements over-confident: stated 0.81, held 0.33).

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

The default `full` registry gives the same results as round 4 after the progressive integration and its rounds 2,
3 and 4 (packs A and E seed 0 re-run on each final code: identical to `reports/round4/runs` key for key, apart from timings).

Tests: full suite 3 052 passed, 4 skipped, 0 failed, 21 min (the evaluator's run on the final round-4 tree, EV-1 – EV-11, sharing the machine with two pack O runs). The APPMON_SLOW Part B test `test_part_b_live_60_equals_900` is left failing on purpose
(60-s t-stream calibration).

## Top open issues (design doc §3.11.1, `progressive.md` §16.12.15–§16.12.16)

1. Recall 0.84 / precision 0.86 (target 0.90): on day 14 most false statements are the drifts of days 12–14
   not yet followed (D1's window, D2's renamed user, D3's renamed pages; precision 0.91–0.98 on actions no drift
   touched, all 5 seeds); on day 21 over-stated `when` coverage on wide windows and sparse group parts. Misses: the
   17:00 report submission on day 14, DEV's ~60-name set (honestly open), the mail departments, FIN's approval
   content on seeds 1–4, the 01:00 backup.
2. Calibration error 0.16 (target 0.05): nodes under-confident on pack O (0.77 stated / 0.89 held), statements
   over-confident on O-real (0.81 / 0.33).
3. D1 confirms after 4–5 workdays on seeds 3–4 (≤ 3 required); D4 one LOW incident on seeds 3–4: the AUTO health
   monitor 192.168.9.9, whose all-zero B24 ring gives a score-0 tie a seeded random p = U — a single-tick alarm with no
   P03 finding behind it (4–7 of ~70 incidents per seed are such conformity-only alarms; lib-3 decision chain owner);
   PG10 day 11 1/5, day 21 3/5.
4. Views: P11's sticky group label keeps the old page name after a rename (`综合部·oa GET /approval/list`); the
   department view lists the old-route parts ("（近期未出现）（页面已更名为 …）") beside the renamed ones; health-monitor
   parts are stated without held-out support. (Fixed in round 4: action lists now name the renamed pages, EV-10, and
   a group's part states only its own members' bindings, EV-11.)
5. PG8 0.83 (finance who arm `prefix` on seeds 1, 3, 4); per-event p95 10–60× the target (P04, P05, P02, P08 hot spots).
6. O-real: 4 of 12 R items, one seed; R13 needs a family rule that respects disjoint user populations.
7. Hash-order dependence in P07 / P02 / P11 value sketches (hidden by the pinned `PYTHONHASHSEED`).
8. Lead decisions: the round-4 gate readings (PG2 pooled debiased calibration, paired trends on the snapshot days,
   PG10 at the stated coverage, EV-9), EV-4's 2 σ cluster tolerance, and the earlier G9 / E10 / value-retention items.
9. Not run: PG9 on packs B–D and seeds 1–4; pack O with the full B library (> 7 GB per run); O-real seeds 1–4. No
   real-log pilot.
10. Statistical library (round-3 list, design doc §3.11.2): 60-s t-stream calibration, gate 14 CPU, the
    identification / class / explanation sub-gates.

## Document map

- `progressive.md` — the progressive core: requirement traceability (§3), data model (§5), algorithms with
  formulas (§6), budgets (§7), engine cards (§8), integration with B01–B30 (§9), bounded mode (§10),
  generator (§11), gates PG1–PG11 (§12), decisions and risks (§14), integration changes and measured
  results (§16; round 2 in §16.9–§16.10, round 3 in §16.11, round 4 in §16.12, status per requirement sentence in §3.1).
- `architecture.md` layered design (§0a: the two parts of library 3); `engines.md` per-engine specs with
  as-built and canonical-mode notes, P00–P15 cards at the end; `cadence.md` the v2.1 dual-grain design
  with implementation notes (§17, §17.1).
- `contract.md` store contract (N: progressive additions); `helpers_api.md` shared maths API.
- `eval.md` gates of packs A–E; `generator.md` personas, packs and scenarios of packs A–E (pack O:
  `progressive.md` §11).
- `integration.md` registry, integration fixes, cost, evaluation rounds (§8–§12), the progressive-core
  integration (§13), its round 2 (§14), round 3 (§15) and round 4 (§16).
- `decisions.md` dropped alternatives and risks; `api_ui.md` API / UI spec.

## History

This file previously held the design-review verdict on the v1 library (stale
carry-forward, re-stamped derived inputs, ungated learning, per-run
IsolationForest, KMeans archetypes, cosine drift and separability, truncated
categorical sets, tick-based thresholds, linear store scans) and the list of
revisions the v2 design made in response. Every item of that list is
implemented; the v1 engines are deleted (integration.md §2) and the
alternatives are recorded in `decisions.md`.
