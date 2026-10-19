# Behavior library (lib-3) — current state

Docs sync of 2026-10-19, updated after round 8 of the progressive core (final code, 5 seeds, O-real 3 seeds, round 7 re-scored with the round-8 scorer: prospective held-out events and PG2's sign test, both lead decisions). The Chinese design document
(`docs/组织业务系统画像平台设计.md`, library 3, §3.0–§3.12) is the full narrative; this page is the
short English summary with pointers. Every number below comes from `reports/progressive/` (pack O) or
`reports/eval_report.json` (packs A–E, round 4; rounds 5–8 re-ran packs A and E seed 0 only — round 5 in `reports/progressive/round5_eval/ae/`, rounds 6–8 in `progressive.md` §16.14.8, §16.15.9 and §16.16.9).

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
  turning into P04 allow-sets or accepted binding values (round 8: an operator's benign verdict absorbs a source into
  P03's who sets, C8-3).

## How the progressive core works (one line each)

| Stage | Method | Engines |
|---|---|---|
| Events | Every request / connection / query (aggregated records expanded through `extra.ev_sample`) becomes an event with all Observation fields, `extra.meta` leaves, parsed body / query key-values and headers; secrets kept as shape; threshold sampling caps learning at e_rate events/s per tree with HT mass; metric-window events carry any fresh raw / derived metric Round 6: a trusted proxy's client is resolved per `ev_sample` row of an aggregated record (S6-6). Round 7: meta leaf names memoised and unchanged values skipped (exact, H7-4). Round 8: a missing lib-4 signature library raises in strict (eval) runs (H8-3). | P00, P01 |
| Open schema | Online type inference; generalisation hierarchies (IP /32 → /24 → /16 → learned group → region → *, route, time, numeric bins, value shapes); roles split / target / invariant / shape / redundant / dropped by measured bits per event with hysteresis Round 6: P05 judges at most 72 attributes per run with rotating invariant / shape / late split roles (T6-5) and per-run caps (T6-8); redundancy is also judged on the rows where the dependent attribute is present (T6-6) and on values for low-cardinality text (EV6-5); compact probe interning (EV6-7); the generalisation memo is keyed on model content and record identity (EV6-8). Round 7: redundancy also judged on the rows where the dependent takes a non-modal value (T7-1); exact duplicates outside the pairwise window are redundant and rotate, plus exact cost rewrites (T7-9). Round 8: when the attribute pool does not fit one P05 run, per-attribute due times with exponential back-off, reset only by a crossing of a role threshold or a level change, hourly runs with nothing due skipped, probe rows following the due slices, node overrides refitted on input changes (T8-2, T8-8, T8-9, T8-11, T8-14, T8-15; pack O unchanged); P02 recomputes hourly statistics only for attributes with new rows and classifies floats without the value memo (exact, H8-4). | P02, P05 |
| Pattern tree | Evidence units ω ≤ 1 per observed row (bursts harmonic); evidence-scaled hierarchical-Dirichlet predictives and prequential code lengths; split rule (V) averaged e-process ≥ 2^(10 + log2 C_ever) (as built: blockwise k-sample e-process), (G) selective MDL gain, (S) time-uniform empirical Bernstein, (D) ≥ 2 dates, (M) ≥ 2 groups × 5 units; route-first partition of the root; value grouping; prune / merge / EFDT revision / budget; single-IP exceptions by a leave-x-out e-value. Round 2: a split is paid only by the behaviour it explains (never by `@who` or source properties such as client stack / TCP window), who levels first and offered as a ladder down to /24, children seeded from the parent's per-value counts, constancy judged over two days. Round 3: split children are born with their own history (≤ 128 replayed rows per learning leaf plus exact per-value extremes), new route nodes get the ≤ 8 rows they waited for, a split never names "not grouped yet" (`grp:∅`) as a child Round 4: every leaf keeps exact per-source and per-/24 extremes from its birth (T-1). Round 5 (cost and determinism, models bit-identical): optional numba kernels of `SplitStats.update` (`lib/pnumba.py`, bit-identical to the numpy path; off without numba or with `APPMON_NO_NUMBA=1`), batched value getter and leaf notes, `TDigest._compress` 14× faster, P05 memos and vectorised `gen_codes`; set summaries counted in sorted order, fixed collapse / pair-sum order, the reference statement identified by a serial number instead of `id()` (T-1 – T-5 of §16.13.2). Round 6: day-type structural coder, an open structural alarm dropped when the coder is replaced (T6-1); request-indexed pair counts and ≤ 48 attributes recorded per row (T6-8, T6-5). Round 7: a new action damped against the route mixture is learned undamped at its own node (T7-5); a released row keeps P03's damping (T7-7); compact `pnode._KC` / `YTable` and `psketch` eviction (exact, T7-9). Round 8: a row waiting at the root counts in its route's burst units (T8-5: the 17:00 report confirmed on day 12); compact extremes and held-out records (T8-18 – T8-20). | P04 |
| Lifecycle and drift | candidate → confirmed (n_c ≥ 20 on ≥ 3 dates) → stable → evolving / stale / retired / dormant; confidence channel (H_l = 30 d, reset on accepted change) separate from shape (H_m = 7 d); ADWIN and Page–Hinkley; coordinated-change acceptance; who-set changes never by persistence alone; trust gating, damping 0.1 for p ≤ 1e-4, held events; daily reference snapshot (dual anchor). Round 2: stated confidence = (passes + 1) / (tests + 2) of held-out tests of everything the statement states (≥ 100 events or a week); sources P03 damped as foreign are suspects and never enter a node's who. Round 3: confirmation on 20 undecayed observations; suspicion is a sequential evidence test cleared only by colleagues' use; the confidence prior is fitted per statement kind (empirical Bayes); B28 keeps learning a source whose open incident concerns one learned pattern only Round 4: a held-out test is ≥ 50 independent source-days or a week, a constraint failing below nominal − max(3 σ binomial, 2 σ cluster-robust); the record forgets per test; a statement takes its kind's current prior at each of its own tests; who is stated at the share its listed sources hold (T-2 – T-6, EV-4, EV-8). Round 5: a structural alarm compares each day with its own day type's pre-alarm loss level (no cross-type fallback, restart on an encoding change) and a transient alarm clears 5 quiet days after its last higher day (EV5-4 a–c). Round 6: days without traffic classed from the calendar (T6-2); numeric drift history per summary scale (T6-3); who-only held-out checks for damped members with colleague support (T6-4); 'when' judged on the trailing day (T6-7). Round 7: the confidence prior pools stated nodes only (T7-2); a restated constraint takes its failed tests with it (T7-3); the `~when` facet (T7-4); a node confirmed without a reference states the pooled base rate (T7-6); rows P03 damped from trusted sources are held-out checks (T7-8). Round 8: an untested statement starts at its kind's first-test pass rate (T8-6); workday and non-workday windows tested separately (T8-7); transport alarms do not withdraw a pattern (T8-13); a restated constraint's failures leave the facet records too (T8-16); an alarm keeps the reference's anchor while the held-out constraints are refreshed daily (T8-22). | P04, B28 |
| Constraints | 90 % bands and exchangeability hard bounds (P(next outside) ≤ 2/(n_rng + 1)) with GPD tails; key sets and character-class grammars by anti-unification, closed value sets, `injection_shape`; approximate FDs with leave-one-out empirical-Bayes Beta lower bounds, set bindings, `cross_binding` / `concurrent_use` / `readdress_candidate`; Bayesian-Blocks activity windows with HDR p; heuristics-miner workflows and required predecessors. Round 3: exact-value capacity grows for finite populations (≤ 64), bindings pool a source's clean days across the tree, route renames are adopted, single-source windows need 5 dates Round 4: a set closes when its Good–Turing unseen mass is below its rarest member's share; windows keep only segments still supported, cross-validate their coverage, do not fit a change younger than 3 dates (marked drift) and state the smallest day-type coverage; bands fit on clean rows; the violation ledger feeds key presence and windows (C-1 – C-6). Round 5: P02 observes per call in batch, P08 screens on integer codes, P06 shares one digest cumsum (bit-identical, R-1 – R-3); set elements sorted (R-4); a source-split child's window reads its ancestor's arrivals for every address its context admits (V-3). Round 6: P09 exact small-sample KS, mean-spacing edge extension, sessions as evidence, all-day for windows that do not hold out of sample, a young-regime width prior (V6-1, V6-2, V6-7, V6-8); P06 leaves out records stating nothing (V6-4); P08 suspends refuted bindings (V6-6); P10 keyed sessions behind a proxy, requirements met by any content variant, rename ledger from the address session (V6-12 – V6-14). Round 7: all-day only on ≥ 5 dates and when the windowed law is no better than a flat day out of sample (V7-1); back-off to the ancestor with the longest dated history (V7-4); identical-output P09 cost work (V7-3); P08 screens once per probe partition and compacts its tables (V7-5); P10 merges a replica family's batches by time (EV7-2); P06's reference p mid-p across ties (EV7-4). Round 8: P10 hysteresis — an accepted edge retired only when the 95 % upper bound of its dependency falls below 0.8, loop edges held on their loop measure, stale edges suspended (V8-1, V8-6, V8-8); a new family key starts from its members' P08 states (V8-3); a multi-valued NAT / proxy source cannot veto a dependency (V8-5); a newcomer value seen at no other source lowers only its own source's bound (V8-9); P06 band ends of discrete attributes snapped to observed values, exact integer back-transform (V8-4). | P06–P10 |
| Who | Weighted MinHash (ICWS) + LSH + mutual-kNN + Louvain over (tree, action) signatures; stable ids; names from `who_group_names` matched on addresses (configuration or inventory import; traffic gives membership, not names); a learned group inside a configured department is one of its roles; prefix covers. Round 3: address pools (configured or learned) are one group covering the prefix; one-address role groups; ids matched on addresses with a lineage Round 4: checked, unchanged (stable from day 5). Round 5: labels and sticky names follow P10's confirmed renames (V-2); covers, Jaccard and modularity sums in sorted order (V-5). Round 6: readdressed members remembered while the old address is silent and stated at the new address (V6-10). | P11 |
| Conformity | Typed p-values who (Good–Turing unseen mass of the closed level) / when / content / seq / novel with back-off and the reference snapshot; Šidák per tick → `behavior.score[conf_*]`; per-day `p_day` for discrete `pattern_violation`; per-IP intensity on guaranteed Space-Saving counts at open nodes. Round 3: a young node's `when` p backs off to its parent; cross-bindings checked up the ancestors Round 4: a source's renamed page is a rename, not a new action; the GPD tail is predictive (parameter uncertainty); improbable transitions are re-scored at route level (G-1 – G-3); B29 explains a pattern violation by the violated constraint (EV-6). Round 5: `HourSS` heap ties broken by push order, not `id(key)` (V-5); a `conf_*` score at its ring's lowest level now gets p = 1 from B24 (lower-atom rule, C-1). Round 6: tree context only for systems with a new batch, per-tick memos, precomputed numeric tail records (`pscore.GpdPredictive`) — exact rewrites (S6-1 – S6-3); the same-batch overlay marks a variant's bare route (S6-7). Round 7: a replica family scored on its shared tree (C7-1); a content-split child never rarer in time than its parent (C7-2); intensity in events, once per tick (C7-3, C7-4); 'when' calibration restarts at an accepted regime change (C7-5); a new action's content not judged against the route mixture (C7-6); a node's first extreme score is not its routine (C7-7); the action ledger per family (C7-8); mid-p across ties (C7-9); a day type's own table only after one complete date (C7-10); `m_ptree._regions` memo (EV7-3). Round 8: an operator's benign verdict absorbs a source into the who sets (C8-3, §6.9.2 (iii)); a cross-tick record cache (exact, C8-4). | P03 |
| Facets and views | Runtime facet registry (10 top facets, 22 default sub-facets); statements with support, confidence, first / last seen, version; system view with per-group / per-department parts, group view and department view (`class:grp:dept:<name>`, composing the department's roles) with negative statements Round 4: duplicates folded, variant conditions stated, configured pools resolved, statements ordered and ending with confidence and support (G-4 – G-6); drifting windows stated as evolving; group parts with their own windows, bands, closed sets and confidence (EV-2, EV-3, EV-5, EV-7, T-7). Round 5: a route-split node is no action (V-1); a department member busy elsewhere is judged on its share within the system (V-4); a system that joined a family has its own view retired (EV5-6). Round 6: a drifting window is stated at the probability that it holds (V6-5). Round 7: a part with its own windows composes the node's record without 'when' (V7-2) with `pwindows.window_hold` (V7-7); the workflow clause contributes its dependency (V7-6); group mass summed in sorted order (EV7-7). Round 8: per-facet confidence (`evidence.facet_conf`); while windows drift the other clauses are stated as a confirmed `~when` facet statement (V8-2), identity facets only for a node P04 never tested (V8-10); held edges rendered (V8-7). | P13, P14 |
| Adaptation and budgets | System characteristics; arms with preconditions; utility in bits per event minus λ·µs; Hedge (one round per completed day) / budgeted UCB; the who arm by held-out behaviour gain (round 2; address code only as a tie-break); system families sharing one tree; P15 water-filling budgets, a 7-step degradation ladder, active and earned sets Round 4: arm costs and budget shares are a deterministic price of counted work (`lib/pcost.py`), families join on the days both systems were observed (E4-1, E4-9). Round 7: PG9 under the lead's decision — bounded mode keeps the earned rule; packs A / E must lose no accuracy and no CPU, the ≤ 0.5 cost target applies to large transient-IP populations over ≥ 48 h (H7-1). Round 8: B14 models created at promotion replay the retained rows and BOCPD is clipped around the current run (C8-1, C8-2); P12 unchanged. | P12, P15 |

## How the statistical library works (one line each)

| Stage | Method | Engines |
|---|---|---|
| Representation | 52 features in 9 groups with explicit exposure, time context (tz, holidays, make-up days), 80-dim token sketch; absence is data | B01 |
| Cadence invariance (spec v2.1, default `grain_mode='canonical'`) | Features on canonical trailing wall-clock grains H = 3600 s and Q = 900 s, built from additive parts, mergeable set sketches and count maps; scored rows only on epoch-aligned decision ticks; midpoint time context | B01, `lib/grains.py`, cadence.md |
| Baselines | Conjugate hierarchical Bayes (entity → role class → system → org → hyperprior, EB pseudo-counts), bin48/bin168 buckets, current (0.1σ15/day) and reference (0.03σ15/day, 24 h delay, golden) anchors, trust-gated delayed reversible commits; Q predictive = native Q anchor + κ_T = 16 pseudo-rows of the H predictive transferred with v = 1 + ((m−1)/m)ω | B03 |
| Exact predictives | NB / Beta-Binomial / Student-t two-sided mid-p against both anchors, p = min(1, 2 min(p_cur, p_ref)); wHMP within and across dependence groups | B04 |
| Detectors | LOO common mode; OAS / C-step T² and SPE with RBC; 15-min slot rhythm (Bernoulli CUSUM, silence, automation index); hierarchical Dirichlet + Good–Turing novelty; client-stack impersonation; PPM-C grammar; timing; Gamma renewal LRT beacon with Monte-Carlo null; POT/GPD budgets; CUSUM / MCUSUM / BOCPD / creep on reference residuals | B05–B14 |
| Identity | Absolute 4-active-H-row windows, OAS-WCCN-LDA with blocked CV (EER_hard, separability), calibrated capped modality LLRs, open-set attribution with other-identity and unknown CUSUMs, Fellegi–Sunter linking and shared-IP tests | B15–B17 |
| Classes as entities | Two-level HDBSCAN hierarchy (role, individual sub-class) plus static CIDR and pool classes; class aggregate baselines and detectors (class_int, class_shape, class_rhythm, class_novel, class_coherence) Round 6: role lineage matched on (old cluster ∪ published members) ∩ eligible (B6-1); bounded mode: members B01 skips are implicit idle rows (B6-3). | B02, B18 |
| Calibration | Randomised Mondrian conformal p per (key, detector, stratum), GPD tail, small-sample prior with randomised pm atoms and an own-history floor, KS / exceedance health. Round 5 lower-atom rule (C-1, C-2): a score at or below the lowest level its ring has seen (empty ring and pm = 1 included) is issued p = 1, the upper p of its tie block, not a seeded coin flip; the fully randomised p is kept (`model.calib['prand']`) for the health monitors and pcal only Round 6: health-weight hysteresis (restore at D ≤ 0.035 and rate ratio in [0.6, 1.67], B6-5). | B24 |
| Fusion | wHMP per family, per-entity meta-calibration with a winsorised tail, single-tick e_day = q_all·n_τ/β_τ over tick types (β = 0.5 / 0.25 / 0.25), one evidence CUSUM per stream (S_t, S_h; ARL 66 d each), accumulator paths, corroborated severities, axis reading rules. Round 5 (C-3): two rows per tick — decisions on the issued row, meta-ring admission / ACI / qcal on the randomised row | B25 |
| Decision and governance | Cadence-invariant decaying risk with fixed L_ref = 60; incidents that close on regime / accumulators / evidence, never on risk; regime machine with legit-vs-attack log-odds, rollback / release / rebase, corroborated REJECT, no permanent lockout; analyst feedback (precision priors, stacking weights, pattern policies, alert budget, label queue) | B23, B26–B28 |
| Explanation and portraits | Natural-unit attribution, faithful counterfactual by deterministic replay of stateful detectors; versioned per-IP and per-class portraits with per-grain bands and diffs | B29, B30 |

## Latest results

### Pack O — the progressive core (`reports/progressive/`, round 8)

`progressive_report.{json,html}`, per seed `runs/O_<seed>.json` (variants `runs/{O-red_0,O60_0,O-real_0,O-real_1,O-real_2}.json`),
PG4 points in `scale_r8/`, PG9 in `pg9_r8/pg9.json`; `progressive.md` §16.16. Code: `code8b` = commit 0ed2de6 in `backend/` and
`scripts/` (the four owners' round-8 work + the evaluator's EV8-1 / EV8-2), `data/` linked. 21 days at 900 s,
`progressive_decision`, `PYTHONHASHSEED=0`, resumable runs, ~43 min and peak RSS ~1.65 GB each with two heavy processes on the
4-core machine (the unchanged lib-3 engines ran at 1.46× their round-7 time: a more loaded machine). **The lead decided two
scoring changes**: held-out claims are judged on the events after the snapshot (days d+1 … d+7 of the segment, prospective)
and PG2's paired confidence by an exact one-sided sign test. "Round 7" below is round 7's runs re-scored with the round-8
scorer (re-run from the published seeds, every JSON reproduced exactly; `reports/progressive/round8_harness/rescore_r7.json`);
round 7's published report and runs are kept in `reports/progressive/round7/`.

| Gate | Target | Round 8 (5 seeds) | Round 7 (re-scored) |
|---|---|---|---|
| PG1 recall @ day 14 | ≥ 0.90 | 0.868 [0.789–0.947] (mean of seeds 0–2 0.912) | 0.868 [0.816–0.947] (0.903) |
| PG1 precision @ day 14, headline (observable ≥ 1 workday) / all claims | ≥ 0.90 / – | **0.955** [0.915–0.981] (pass) / 0.883 [0.848–0.912] | 0.980 [0.938–1.000] / 0.884 |
| PG1 components who / when / content / bindings / workflow | ≥ 0.85 each | 0.947 / 0.974 / 0.921 / 1.00 / 1.00 | 0.921 / 0.947 / 0.921 / 1.00 / 1.00 |
| PG1 GA + FIN 6 bindings at day 14 | 6/6 | **5/5** (P14 states a drifting window's other clauses on their own) | 4/5 |
| PG1 recall / precision @ day 21 | – | 0.881 / **0.985** | 0.881 / 0.971 |
| PG1 recall / precision (observable) over days 5 / 10 / 14 / 18 / 21 | rising | 0.43 / 0.80 / 0.87 / 0.93 / 0.88; 0.89 / 0.94 / 0.95 / 0.97 / 0.99 | 0.44 / 0.78 / 0.87 / 0.90 / 0.88; 0.96 / 0.94 / 0.98 / 0.96 / 0.97 |
| PG2 calibration error (debiased, days ≥ 7 pooled); median confidence day 7 → 21; paired confidence non-decreasing (sign test); depth non-decreasing | ≤ 0.05; rising; 5/5; 5/5 | 0.059 [0.049–0.076] (stated 0.942, held 0.941; seed 2 0.049); 0.87 → 0.97; **4/5** (seed 4: 23 lost / 4 gained, p = 0.0002); 5/5 | 0.059 (0.940 / 0.944); 0.88 → 0.98; 5/5; 5/5 |
| PG3 GA login = 3 IPs / finance approval = 1 IP / portal at prefix / DEV pool one group / ARI | yes / ≥ 0.9 | 5/5 / 5/5 / 5/5 / 5/5 / 0.974 (pass) | same |
| PG4 memory vs IPs / attributes / systems; CPU / event vs IPs / attributes; per-event p95 scoring / learning | ≤ 0.15 / 0.2 / 0.3; ≤ 0.1 / 0.2; ≤ 100 / 250 µs | 0.016 / 0.194 (pass) / 0.251; −0.066 / **0.206**; 1.16 / 3.29 ms (host 1.46× more loaded) | 0.015 / 0.187 / 0.258; −0.057 / 0.260; 0.72 / 1.98 ms |
| PG5 D1 / D2 / D3 / D4 / D5 / .21 still `jack` | yes | 3/5 / 5/5 / 5/5 / 5/5 / 4/5 / 5/5 | same |
| PG6 anomalies A1–A10; FAR ≥ LOW / ≥ MEDIUM; pattern violations ≥ LOW; B29 top reason | ≥ 0.95; ≤ 0.10 / ≤ 0.02; ≤ 0.05; ≥ 0.9 | 50/50; 0.0013 / 0.0005; 0.0058; 0.92 (gate passes) | 50/50; 0.0014 / 0.0005; 0.0053; 0.92 |
| PG7 registered / role ≤ 24 h / type correct | 1 / 1 / ≥ 0.95 | 1 / 1 / 1.0 (pass) | same |
| PG8 arms in truth; switches after day 7; who within 5 % | ≥ 0.95; ≤ 2; yes | 0.833; ≤ 1; 2/5 | same |
| PG9 | A / E: no accuracy loss, lib-3 CPU p95 ≤ full; large transient populations: p95 ≤ 0.5 × full | fails pack A gate 3 only (bounded 8 incidents ≥ MEDIUM vs full 7; now the `intensity` family re-opening one incident, B14's alarm is gone); A / E CPU p95 0.81 / 0.81; 1 000 transient IPs over 84 h 0.43 | same failing check (B14's burn-in); 0.88 / 0.77; 0.47 |
| PG10 finance single IP / GA negative / OA statement day 11 / day 21 | all | 5/5 / 5/5 / 1/5 / 3/5 | same |
| PG11 (O-real, 3 seeds) | ≥ 4/5 per R item | R8, R10 3/3; R7, R9 2/3; R4, R11 1/3; others 0 (5 / 4 / 3 per seed) | 4 / 4 / 3 per seed |
| Example checklist (day 21) | 50 | 47 | 47 |

What round 8 changed: the profile no longer drops what did not change while something else drifts (GA + FIN bindings 5/5;
P10 holds an edge sitting at its cut), a borrowed credential is caught on the replica family of O-real for the first time
(A2 on every seed: the family's binding state is inherited from its members), and P05 no longer re-judges every attribute
every hour at 340 attributes (CPU slope 0.260 → 0.206, P05's growth 2.82× → 1.82×). Recall and calibration are round 7's
at the median; day-21 precision rose to 0.985 and seed 2's calibration error reached 0.049.

Cost (pack O, 21 days, median): P-core 1 202 s wall / 1 184 s thread CPU against round 7's 723 s; with the load factor 1.46
that is 1.14× at equal load (P06 1.50×, not profiled); peak RSS 1.50 → 1.65 GB. Determinism: pack O seed 0 under
`PYTHONHASHSEED` 0 and 1 — 0 of 19 350 score leaves differ and the full result pickle is identical.

The requirement's example as learned (design doc §3.8.0a quotes P14's statements verbatim, both views, pack O days 11 / 14 / 21
and O-real day 35; also `progressive.md` §16.16.10): 47 of 50 checklist clauses pass. Seed 0's day-21 login statement is
round 7's in every clause: "工作日 08:31–08:51（覆盖 89 %，6 个工作日），综合部（10.168.7.121、192.168.1.21、192.168.1.23）访问 POST /login（登录）：…
username= 取值 `[a-z]{4}(\.[a-z])?`，username= 取值集合封闭 {jack, mike, mike.w, rose}；…绑定：10.168.7.121 → username=mike.w、192.168.1.21 → username=jack、
192.168.1.23 → username=rose…。置信 0.97 · 依据 42 次". New on seed 2, day 14 (D1 moving the window): the windowed statement is
`evolving` at 0.02 with "时间窗正在变化：时间置信 0.02，其余约束置信 0.60", and a confirmed facet statement states who, the three
bindings and the login → home flow at 0.92 ("内容约束尚未经留出数据检验，未单独陈述"). Misses: seed 1's 90 % band (1–2.5 KB),
seed 3's band (1–3 KB) and seed 4's login window (IoU 0.68).

Variants: O-red seed 0 (never tuned on) recall @ 14 0.730 (round 7 0.649), anomalies 10/10, calibration error 0.162 (0.244;
still under-confident), **D2 fails** (in P05's due regime the username is dropped on a no-login Saturday and P08 loses the
login node's bindings, so 10.77.2.19's rename is never stated); O60 seed 0 day-8 agreement 0.047 (inside 0.05). O-real
seeds 0–2 (35 days): **A2 detected on every seed**, anomalies 10 / 9 / 10 (seed 1 misses A5), bindings @ 14 3 / 0 / 3 of 6
(0 / 0 / 0), recall @ 35 0.63 / 0.72 / 0.61, calibration error 0.116 / 0.082 / 0.098 (0.109 / 0.094 / 0.116), FAR ≥ MEDIUM
0.0007 / 0.0012 / 0.0010, checklist on day 35 5 / 6 / 7 of 10 (7 / 6 / 7: on seed 0 the family login node holding
综合部's .121 / .23 and finance's .2.12 / .2.51 is stated at /24 level; the 综合部 login — 08:29–08:50, {mike.w, rose}, the
bindings .121 → mike.w and .23 → rose — appears only in group G19's user view; bisected to T8-5).

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

Rounds 5 and 6 changed the shared decision chain (round 5: the B24 / B25 lower-atom rule; round 6: B02 role lineage,
B11 / B12 prefilter, B18 implicit idle rows, B24 health-weight hysteresis), so the default registry is no longer identical
to round 4. Packs A and E seed 0 were re-run on each final code (round 5: `reports/progressive/round5_eval/ae/`; rounds 6 and
7: the evaluator's scratch runs, not committed; `progressive.md` §16.14.8, §16.15.9). Round 7 changed no B engine and was
**identical to round 6**: A 12/12, 8 within deadline, control incidents 18 / 7 / 2, reopened 8, class ARI 0.780, L8 low;
E 3/3, 6 / 1 / 0, class ARI 0.755, refit ARI minimum 0.955; gate 7's anti-conservative D⁺ max 0.102 (A) and 0.129 (E).
The round-7 lib-4 change (`signature.rule_match` / `correlation` evaluate only entities that can match this tick) leaves
these runs unchanged. A first round-7 run showed pack A at 12 / 7 / 1 only because the evaluator's code snapshot lacked
`data/` (no signature library); re-run with `data/` linked. Bounded mode: gate 15 passes (round 6); PG9 under the lead's
round-7 definition fails only pack A's gate 3 (one extra MEDIUM: B14's 3-h burn-in after P15 promotes an IP), with lib-3 CPU
p95 at 0.88 (A) and 0.77 (E) of full mode and 0.47 on the 1 000-IP, 84-h point. `eval_report.json` was not regenerated
(packs B–D and seeds ≥ 1 not re-run). Round 8 changed only B14 (C8-1, C8-2): pack A full mode is round 7's (12/12, 18 / 7 / 2,
reopened 6), pack E has the same incidents with class ARI 0.797 (0.755; not traced); PG9 fails only pack A's gate 3 (the
`intensity` family re-opens one incident at MEDIUM in bounded mode), lib-3 CPU p95 0.81 / 0.81, the 1 000-IP point 0.43.

Tests: full suite on `code8b` 3 397 passed, 4 skipped, 1 failed (37 min, beside a scale point): `test_b13_budget_edges::
test_perf_forty_entities_generous_bound` at 83 ms against 80 ms (77.9 / 78.7 ms alone; B13 unchanged since round 7; bound
not loosened). Round 7: 3 319 passed, 0 failed. The APPMON_SLOW Part B test `test_part_b_live_60_equals_900` is left failing on purpose (60-s t-stream
calibration).

## Top open issues (design doc §3.11.1, `progressive.md` §16.16.11–§16.16.12, §16.16.14)

1. Recall @ 14 0.868 (target 0.90; mean of seeds 0–2 0.912): every remaining day-14 miss on seeds 0–2 belongs to P06 /
   P07 / P09 / P14 — DEV's 60-value username set and the FIN approval opinion field (P07, OI8-b); one department across
   sibling nodes or split by time of day (P14, OI8-a); windows and bands fitted from ~16–20 arrivals too narrow (P09 / P06,
   OI8-f / OI8-h); day 21: AUTO.backup#0's 02:00 window on every seed.
2. PG2: untested nodes state the kind's pooled prior, refitted daily, so all of them move together and the sign test fails
   (seed 4, O-real seed 2, O-red): hold an untested node at its stated value until its first test, or the lead decides that
   PG2 pairs only tested statements. Calibration error 0.059 vs 0.05 (seed 0: four day-7 claims at 0.00–0.055, OI8-j;
   O-red 0.162 under-confident on the P04 / P14 side).
3. O-red D2 (new): P08 must keep a stated binding's payload among its candidates, keep its node and its value history
   (patch proposed for `binding.fit_tree`); P05 should not demote a role on a day with no rows where the attribute is
   present, and keep the target role of a bound payload; P05's node overrides admit independent noise attributes.
4. O-real: checklist 7 → 5 on seed 0 (T8-5 splits the family login node by /24 on day 18); configured departments as parts
   on shared family nodes (OI8-g, ~20 of seed 0's 30 day-35 misses fail on who only); seed 1's A5 (the requirement is never
   accepted); A2's source stays damped after its incidents (trust per (source, credential) pair proposed).
5. PG4: CPU / event slope vs attributes 0.206 (at the limit, within the measurement's own noise); P02's slope follows
   P05's kept count (OI-H2); per-event p95 10× the budgets.
6. PG5: D1 on seeds 3–4 (sample size); D5 from P00's record-mean packet counts — the fix adds false MEDIUM incidents
   through P03's sequence path, which must be explained first (EV8-3).
7. PG9: the B-library `intensity` family re-opens pack A's inc0000037 at MEDIUM in bounded mode (OC8-1: replay the
   retained rows at promotion); per-tick loops over every known entity in D2 session, B21, B17, B27, B23 and
   `MetricStore.entities_active` (OI-H1, patches proposed).
8. PG8 0.833; PG10 day 11 1/5, day 21 3/5; O-real 3–5 of 12 R items per seed (R12, R13, R1–R3, R5 fail everywhere).
9. Evaluation: packs B–D and seeds ≥ 1 of A / E not re-run since round 4; pack O with the full B library (> 7 GB per run)
   not run; B13's timing test borderline on a loaded host; prospective windows over a declared holiday (lead's call). No
   real-log pilot.
10. Statistical library (design doc §3.11.2): 60-s t-stream calibration, gate 14 CPU, the identification / class /
    explanation sub-gates.

## Document map

- `progressive.md` — the progressive core: requirement traceability (§3), data model (§5), algorithms with
  formulas (§6), budgets (§7), engine cards (§8), integration with B01–B30 (§9), bounded mode (§10),
  generator (§11), gates PG1–PG11 (§12), decisions and risks (§14), integration changes and measured
  results (§16; round 2 in §16.9–§16.10, round 3 in §16.11, round 4 in §16.12, round 5 in §16.13, round 6 in §16.14, round 7 in §16.15, round 8 in §16.16, status per requirement sentence in §3.1).
- `architecture.md` layered design (§0a: the two parts of library 3); `engines.md` per-engine specs with
  as-built and canonical-mode notes, P00–P15 cards at the end; `cadence.md` the v2.1 dual-grain design
  with implementation notes (§17, §17.1).
- `contract.md` store contract (N: progressive additions); `helpers_api.md` shared maths API.
- `eval.md` gates of packs A–E and the progressive core's readings per round (PG9's round-7 definition; round 8's sign test and prospective held-out events); `generator.md` personas, packs and scenarios of packs A–E (pack O:
  `progressive.md` §11).
- `integration.md` registry, integration fixes, cost, evaluation rounds (§8–§12), the progressive-core
  integration (§13), its round 2 (§14), round 3 (§15), round 4 (§16), round 5 (§17), round 6 (§18), round 7 (§19) and round 8 (§20).
- `decisions.md` dropped alternatives and risks; `api_ui.md` API / UI spec.

## History

This file previously held the design-review verdict on the v1 library (stale
carry-forward, re-stamped derived inputs, ungated learning, per-run
IsolationForest, KMeans archetypes, cosine drift and separability, truncated
categorical sets, tick-based thresholds, linear store scans) and the list of
revisions the v2 design made in response. Every item of that list is
implemented; the v1 engines are deleted (integration.md §2) and the
alternatives are recorded in `decisions.md`.
