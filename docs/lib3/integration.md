# lib-3 integration record

What the integrator did with the 23 engine reports in
`engine_integration_notes.json` (94 requested shared changes plus the listed
deviations), how the full v2 registry is assembled, what was retired, the
end-to-end results in strict mode, the per-engine cost, and what is still open.
Every change below is covered by a test: `tests/test_integration_shared.py`
(shared code and cross-engine contracts), `tests/test_pipeline_e2e.py` (the
mini pack end to end), and the engine test files named in the rows.

## 1. Registry (backend/app/pipeline/build.py)

`build_registry()` registers every v2 engine in architecture §1 order. Within
a layer the engines run in registration order.

| Layer | Order |
|---|---|
| raw | l2l3, l4flow, http, tls, dns, active_probe, action_token (R2), client_stack (R3) |
| derived | aggregation, periodicity, trend (D0); ratio, entropy, graph (D1); session (D2) |
| behavior | B01 feature_vector, B02 peer_group, B03 baseline, B04 likelihood, B05 common_mode, B06 multivariate, B07 rhythm, B08 novelty, B09 client_identity, B10 sequence, B11 timing, B12 beacon, B13 budget, B14 changepoint, B15 identity_model, B16 attribution, B17 entity_link, B18 class_monitor, B21 cross_system (P2, round 4; `build_registry(p2=False)` leaves it out), B23 feedback, B24 calibration, B25 fusion, B26 risk, B27 incident, B28 governor, [B29 explain slot: `_explain_engines()`], B30 portrait |
| signature | rule_match, correlation |

Same-tick dependencies that this order satisfies (producer before consumer):
B01's `feature.active` is the clock of every learner and of B11; B04 writes
`behavior.z` before B05 reads it; B05 writes `behavior.zi` and
`behavior.common.<g>` before B06, B18 and B25 read them; R3 and B01 run before
B09; B24's `behavior.p` comes before B25, B25 before B26, B26 before B27, and
B27 before B28.

One-tick lags that the spec allows and the engines are written for:
- B05 reads B08's novelty events of t−1, because B08 runs after it.
- Every learner reads B28's trust and quarantine of t−1. Learners commit row t−D.
- B26, B27 and B28 read lib-4 matches with a one-tick lag, since signature runs last.

P2 engines B19, B20 and B22 are not registered; B21 cross_system is (round 4, §12). They stay behind the ablation gate.

Runtime (`Runtime`): `ctx.config` carries tz, the calendar (holidays and
make-up workdays from the generator clock) and `strict`. Warm-up runs with
`training=True` over the warm-up plan (spec v2.1 default `Runtime.DEFAULT_WARMUP_PLAN`
= 120 × 3600 s + 192 × 900 s; `warmup_ticks=n` keeps the v2 plan n × 900 s). The live loop runs at the runtime window. Every
tick is stamped `now = gen.vt`. `limit_blas_threads(1)` pins BLAS and OpenMP to
one thread (R15.1, R19.3).

The eval seam is unchanged: `eval/runner.default_registry_factory` calls
`build.build_registry(config=…, pack=…, seed=…)`. `run_pack(…,
registry_factory=…)` still accepts any `callable(pack=?, config=?, seed=?)`.

## 2. v1 retirement

- Deleted: `engines/behavior/{anomaly,drift,fingerprint,clustering,features,util_norm}.py`.
- `tests/test_helpers_sanity.py` was ported to v2. It now writes `feature.nat`
  and `feature.active` rings and asserts B03's `model.baseline` and
  `profile.baseline_median`. It no longer checks `profile.stable`, which B01 owns.
- The `read_feature_matrix` shims are removed; nothing references them.
- `api/routes.py` got a minimal v2 projection, so the API imports and every
  existing endpoint responds (all route functions were exercised on a warmed
  v2 runtime):
  - `anomaly_score` comes from `behavior.e_day`, else `q_all`, mapped to [0, 1] as −log10(·)/6.
  - `drift_score` comes from the strongest of `behavior.p` (else `pm`) over cusum, mcusum, bocpd and creep.
  - `archetype` is `profile.archetype`, which is B02's class path.
  - `risk` is `behavior.risk`.
  - `entity_detail` reads z from `behavior.z`.
  - W7's API v2 (`api/routes_v2.py`, `api/views.py`) has since been merged on
    top. After the merge the legacy route functions were re-exercised on a
    warmed v2 runtime and `tests/api` passes.
- `scripts/smoke.py` runs the runtime (warm-up over the Runtime plan, then live 60 s) and
  prints per-engine timings, class paths, separability, the risk top-10,
  incidents and recent events. It exits non-zero on an engine error.

## 3. Requested changes: triage

Status values: **applied**, **docs** (applied as documentation), **rejected**
(with a reason), **deferred** (optional or owner-side, with a reason), and
**note** (no action requested).

### D0 / D1 / D2
| # | Request | Status |
|---|---|---|
| R0.0 | Retention of the trend targets (13 h) and the D0 inputs (24 h) | applied: `store.DEFAULT_RETENTION` and the contract B table |
| R0.1 | engines.md D0: activity gate, rescaling, Welch z, beacon_lag = 0, act.events | docs |
| R0.2 | fresh.py could adopt the vectorised `grid_many` | deferred: optional perf. `fresh.grid` stays the reference implementation, D0's `grid_many` is tested equal to it, and D2 does not grid per metric |
| R1.0 | Meaning of `derived.<x>_n` (includes `__other__`; entropy excludes it) | docs (contract B) |
| R1.1 | Graph peers include `l4.peer_set` buckets at eTLD+1 | docs (contract B) |
| R2.0 | D2 after D0 and D1 | applied (build.py) |
| R2.1 | D2's one-time `set_retention` could lower `act.events` | applied: new `MetricStore.ensure_retention` (raise-only), used by D0 and D2. B18, B24 and B25 use it for their own series too. B13 keeps its explicit points cap, which is a memory bound on its own output |
| R2.2 | engines.md D2: req_per_session absent, duty time-weighted, sessions not reweighted | docs |

### B03 / B04 / B06
| # | Request | Status |
|---|---|---|
| R7.0 | B03 at interval 1 | applied |
| R7.1 | `test_helpers_sanity` encodes v1 | applied: ported to v2 rings |
| R7.2 | `model.baseline.slope_log` for B28 | rejected: not in contract C and no consumer needs it. B28 takes the ramp slope from B14's `baseline_creep`, else its own Sen slope. Contract C now says there is no slope_log field |
| R7.3 | Document m_baseline and its choices | docs: engines.md B03, contract C, helpers_api `m_baseline` |
| R7.4 | B04 on `predictive_set` / `midp` / `quantiles` / `values_from_nat` | applied (B04 uses `predictive_set`; `state_quantiles` delegates to `MB.quantiles` since R16.0) |
| R7.5 | Eval snapshots: anchor summaries, not gate journals | applied: `runner._baseline_snapshot` |
| R16.0 | **Bug**: `quantiles` / `mean_nat` applied the t-family inverse transform on NIG-block positions of the full row | applied: `_inverse_tx` works on the NIG block, and `_inverse_tx_full` covers full rows (used by `descriptors`). Test: `test_quantiles_and_mean_nat_invert_t_family_on_the_right_columns` |
| R16.1 | `quantiles` calls `bb_ppf` once per (q, feature) | applied: one grouped `bb_ppf` call |
| R16.2 | `m_class.members` / `class_key` O(N) per call | applied: per-(model, version) member index (`m_class._Index`, copy-on-write). Test: `test_m_class_index_follows_copy_on_write_versions` |
| R16.3 | Public scalar BB triple | applied: `bayes.bb_parts`. B04 keeps its lgamma pmf, which is cheaper for the pmf alone |
| R16.4 | Register B04 after B03 | applied |
| R15.0 | Register B06 in place of AnomalyEngine | applied |
| R15.1 | Single-threaded BLAS | applied: `tests/conftest.py`, `build.limit_blas_threads`, and the smoke and evaluate scripts |
| R15.2 | `c_step_oas` shrinks strong pairs | note: awareness only. The null calibration is fine |
| R15.3 | B04: `m_density.groups(split_by_feature_group=True)`; B29: `score_model` / `contributions_model` | applied for B04. The B29 part waits for B29 |
| R15.4 | Tie-mass zi columns excluded from B06 | note (documented by B06) |

### B05 / B08 / B26 / B27 / B28 (decision spine)
| # | Request | Status |
|---|---|---|
| R5.0 | Register B26 after B25 and before B27 | applied |
| R5.1 | `store.timeline` lists risk from the vec ring | applied: points are listed on 10-point band changes. Test: `test_timeline_lists_risk_from_the_vec_ring` |
| R5.2 / R6.0 | B08 events carry `extra.tier`, `adopted`, `dim` / `value`, and flags | applied: `extra` {dim, value, tier, bits, idf, adopted, discount, flags{…}} with the flags also as top-level keys. B05 applies the system-tier exclusion at t−1; B26 reads weight, repeat key and stage |
| R5.3 | B27 / B29 read `behavior.risk` as a vec ring | applied (B27 `vec_at` / `vec_tail`). B29 waits for B29 |
| R5.4 | Contract: `behavior.risk` as a scalar ring; lowercase tiers | docs (contract B) |
| R6.1 | Persistence of B05's loading state | deferred: B05 owns no contract-C model. The state lives in the engine with gating checkpoints, and a restart falls back to β = 1, the documented safe prior |
| R6.2 | Document the `behavior.common.<g>` layout for B18 and B25 | docs (contract B). Verified: B18 reads `dir` / `frac_up` / `run`, and B25 keys its caps on the same axis names |
| R6.3 | B28: quarantine = 1 at or before the tick of `rollback_to` | applied: `_machine` announces SUSPECT and writes `rollback_to`, then `_quarantine` is computed on the same tick. Learners see quarantine(t−1) = 1 together with the directive |
| R13.0 | Register B27 after B26 and before B28 | applied |
| R13.1 | `suppressed_common` status / quiet close_reason | docs (contract E): status `suppressed` + `parent_id` (evidence state `suppressed_common`); the quiet close uses `timeout` |
| R13.2 | Shared accumulator level | applied: `detectors.acc_level(p, d, dt)` = ln(1/p)/ln(ARL_ticks). B27's `acc_level_from_p` delegates to it and B28 uses the same scale, vectorised. Test: `test_acc_level_scale` |
| R13.3 | `behavior.regime` dict with `state`; regime events with `extra.state` | applied (B28) |
| R13.4 | `model.link` accessor | applied: `lib/m_link` (links, aliases, retractions, actors, continuity, shared_ip). B27 still parses the same layout itself (identical semantics); switching is optional (R21.3) |
| R14.0 | Register B28 after B27, with a B29 slot | applied |
| R14.1 | Regime types `c2` and `exfil` | docs (contract F) |
| R14.2 | B03 slope diagnostics | rejected (see R7.2) |
| R14.3 | B23 queues the governor label queue (`reason 'held'`) | applied. Test: `test_governor_label_queue_keys_are_queued_by_b23` |
| R14.4 | B27's quiet test on the raw `m_cp.level` almost never passes | applied: B27 and B28 use the calibrated-p `acc_level`; the `m_cp.level` docstring says it is diagnostic only |
| R14.5 | engines.md B28 notes | docs |

### B07 – B14
| # | Request | Status |
|---|---|---|
| R3.0 | engines.md B11 test (a) wording; B and M gaps < 30 min | docs |
| R3.1 | B01 ahead of B11 | applied (registry; e2e asserts the order) |
| R3.2 | B15, B16 and B30 consume `m_timing` accessors | applied: m_identity uses `m_timing.loglik` / `pmf` / `bin_index`, and B30 uses `m_timing` descriptors |
| R8.0 | Register B08 after B07 and before B10 | applied |
| R8.1 | B12 on `m_vocab.dest_prevalence` | applied (R2's counts are the fallback) |
| R8.2 | helpers_api `m_vocab` | docs |
| R8.3 | Consumers of m_vocab (B13, B18, B02, B15–B17) | note: they use `dest_prevalence`, `adoption_records`, `family_distribution` and `loglik_models` |
| R9.0 | Register B09 after B08 (after R3 and B01; B28 after it) | applied |
| R9.1 | engines.md B09 deviations | docs |
| R9.2 | Contract C: `model.client@__system__` tiers, cooc, acq / known | docs |
| R9.3 | helpers_api `m_client` | docs |
| R11.0 | Register B12 | applied |
| R11.1 | `m_vocab.dest_prevalence` | applied: `dest_prevalence(store, s, name, now=None)` |
| R11.2 | engines.md B12 (κ0 = 1.5 null, rank size test, Z² band) | docs |
| R11.3 | `read_dict` looks back 4 points; `get_model` returns live objects | note for test authors |
| R12.0 | Register B13 after B12 and before B14 | applied |
| R12.1 | Retention of `behavior.budget` | applied (store default: 24 points and 1 d; contract B) |
| R12.2 | Config `budget_abs_floor` | applied: `DEFAULT_CONFIG` and contract I, with the default floors listed |
| R12.3 | engines.md B13 (dns.qname_set, level-q guard, log space, floors) | docs |
| R12.4 | Keep the `dest_prevalence` signature | applied (unchanged) |
| R12.5 | Exact shape of `model.link.actors` | docs: contract C gives actors `{id, members, links, first_ts, last_ts}`; `m_link.actor_members` |

### B02 / B15 – B18 / B30 (identity and classes)
| # | Request | Status |
|---|---|---|
| R18.0 | Register B02 in place of ClusteringEngine | applied |
| R18.1 | engines.md B02 (refit stride, ε merge, d90 floor) | docs |
| R18.2 | Document the extra fields of `model.class` | docs (m_class docstring, contract C) |
| R19.0 | Register B15 in place of FingerprintEngine | applied (v1 retired) |
| R19.1 | engines.md B15 reads and interval; contract C `model.identity` | docs |
| R19.2 | How B16 and B17 score windows | note (B16 and B17 follow it) |
| R19.3 | Single-threaded BLAS | applied (see R15.1) |
| R20.0 | Fast `window_vector` into m_identity | applied: `m_identity.window_vector` uses the one-sort quantile path, and B16 delegates to it |
| R20.1 | Vocab factor inside `modality_logliks` | applied (`m_identity.vocab_factor`), so B15's `llr_calib['vocab']` is fitted on the scaled LLR |
| R20.2 | Drop the client.stack_set sketch block from the window vector | applied: `W_SK_CLIENT` is zeroed. This exposed a B16 issue: partial windows (fewer than K rows, as on a new or long-silent entity's first ticks) failed the χ² typicality and raised `unknown_identity` on enrolled entities. B16's unknown CUSUM now only climbs on full K-row windows. Tests: `test_window_vector_zeroes_the_client_stack_sketch_block` and B16 (a) and (b) |
| R20.3 | Move B16's per-tick caches into `m_identity.Background` | deferred: perf-only. B16's caches are tested equal, and moving them changes B15's public API, which is for the owner (B15) to do |
| R20.4 | `m_link.aliases` / shared-IP | applied: `m_link.aliases(model, e)`, `shared_ip(model, e)` and `continuity(model, e)` take the model from `m_link.get(store, s)`. Retracted links are excluded |
| R21.0 | B28 answers a retracted link with rollback + release | applied: `GovernorEngine._link_retractions`, once per link and deferred while a rollback is rate-limited. Test: `test_retracted_link_rolls_the_seeded_entity_back_and_releases_it` |
| R21.1 | Register B17 | applied |
| R21.2 | B02 skips retracted links | applied (`m_link.links(active_only=True)`) |
| R21.3 | B13, B16 and B27 switch to m_link accessors | deferred: optional. Their parsers match the layout, and switching is a refactor with no behaviour change |
| R21.4 | Document `model.link` and index m_link | docs (contract C, helpers_api `m_link`, engines.md B17) |
| R21.5 | A warm-up link can be retracted later | note: accepted known risk. B28 undoes a retraction exactly (rollback to `t_link`, then release), so the cost is compute only. B17 links in training by design (its edge test) |
| R17.0 | Retention of `behavior.class.agg` (9 d) | applied (store default) |
| R17.1 | Adoption ledgers for static and pool classes | deferred: optional. B18 reads the members' role-class ledgers |
| R17.2 | Keep the class axis names stable | applied (unchanged; they match fusion's `canonical_axis` and `COMMON_MODE_AXES`) |
| R17.3 | engines.md B18 perf line | docs |
| R22.0 | Register B30 after B28 | applied |
| R22.1 | `m_baseline.n_eff_by_bucket` | applied. Test: `test_own_support_and_n_eff_by_bucket_of_an_empty_model` |
| R22.2 | Automation index A in `model.class.assign` | applied (`assign.A`; B30 prefers it) |
| R22.3 | Portraits read aliases and actors through m_link | note: B17 writes `profile.extra.continuity` from `m_link.continuity`, which is what B30 reads |

### B23 – B25
| # | Request | Status |
|---|---|---|
| R4.0 | B24 detects a version change through `gate.version` | applied. Test: `test_b24_resets_rings_on_a_bare_version_change_from_the_default` |
| R4.1 | Retention of `behavior.e_day` (8 d) | applied (store default; contract B) |
| R4.2 | engines.md B25 test (c) over replicates | docs |

Totals:

| Status | Count |
|---|---|
| applied (code) | 56 (R5.3 and R15.3 cover only B27 and B04; their B29 parts wait for B29) |
| docs | 24 |
| rejected | 2 (R7.2, R14.2: `slope_log` has no contract-C field and no consumer needs it) |
| deferred | 5 (R0.2, R6.1, R17.1, R20.3, R21.3) |
| note | 7 (R8.3, R11.3, R15.2, R15.4, R19.2, R21.5, R22.3) |
| total | 94 |

## 4. Cross-engine contract fixes found end to end

Each fix is at its root cause and has a test.

| Problem | Root cause | Fix |
|---|---|---|
| On the first live day every entity alarmed (new_template_ratio z ≈ −4 to −8 for hours; mini pack) | R2 wrote `act.new_template_ratio = 1.0` on the first tick of an empty vocabulary. B03 learnt it into the buckets of that day type, and the first live day of the same type was scored against it | R2 writes the ratio only once `model.template` is ≥ 24 h old (`m_template.NEW_REF_S`, `model['born']`). Before one daily cycle, "new to the system" measures the vocabulary's own growth. Tests: `test_new_template_ratio_waits_for_one_daily_cycle`, `test_new_template_ratio_needs_a_mature_system_vocabulary` |
| Every busy entity sat at risk 40–60 with credential / privilege stages in the production runtime | lib-4 severities grade activities (routine login, form write, admin page and poor TCP are `low`), and B26 weighted every low match 5 with the auth / admin / transfer categories as stages. The spec's null (L ≈ 7) did not include lib-4 | Habitual lib-4 activity weighs 0 in B26: a match of severity ≤ medium whose (entity, signature) has ≥ 4 matched ticks, the first ≥ 24 h earlier. A new activity counts for its first day; high and critical always count. Test: `test_habitual_routine_lib4_activity_stops_counting` |
| `unknown_identity` on enrolled entities at their first ticks | B16 scored partial windows (fewer than K rows) with the χ² typicality calibrated on B15's K-row windows | The unknown CUSUM only climbs on full windows (B16 tests (a) and (b)) |
| Two engines set the same input retention in registry order (R2.1) | `set_retention` overwrites | `store.ensure_retention` is raise-only. Test: `test_ensure_retention_only_raises` |
| lib-4 `rule_match` cost ~13 ms per tick at 20 entities | `store.snapshot` scanned every raw and derived name, and lib-3 adds ~300 derived names and vector views per entity | `snapshot(…, names=…)`: rule_match asks only for the metrics its signatures reference. Same values; 13 → ~4 ms. Test: `test_snapshot_restricted_to_names_equals_the_full_snapshot` |
| B30 cost 60–80 ms bursts every 8th tick | An 8-tick engine stride refreshed every key on the same tick | B30 runs every tick and refreshes each key once per 2 h at its own phase (`entity_due`). Same work, spread (B30 scheduling test) |
| B15's live calibration cost ~7 ms per system per tick | Three entities sampled PPM and vocab LLRs under 4 candidates on every tick | An entity opens its next calibration window ≥ 4 h after its last (`CAL_GAP_S`). About 6 windows per entity per day, and MIN_CAL is still reached within hours |
| (eval) lib-4 bulk_upload (HIGH) matched a nightly backup host on ~150 of 152 ticks, most of them idle; its warm-up trust was 0, B28 REJECTed and froze it on the first live tick | `rule_match` took `store.snapshot` without `now`, i.e. the latest value of any age | Fresh-only snapshot (`now=ctx.now`): a match describes this tick. Test: `tests/test_rule_match_fresh.py` |
| (eval) `periodicity` z ≈ 8–30 on 87 % of pack A's live rows | D0 wrote periodicity_score = 0 when the 6-h grid holds fewer than MIN_TICKS = 8 points (every 3600-s tick), so every workday bucket learnt a point mass at 0 | Undefined, not 0: no periodicity_score / beacon_lag is written below MIN_TICKS bins (B01 reads NaN). Test: `test_periodicity_undefined_when_the_span_holds_too_few_ticks` |
| (eval) identity p = 1e-38 on the first 900-s tick of every enrolled entity after a 3600-s warm-up | B24's identity stratum was (daypart, regime tercile) without the cadence class; B16's K-tick windows make pi_self far more certain at 3600 s, so the 900-s score hit the 3600-s ring's GPD tail | Identity stratum = (daypart, tercile, cadence class) (`calib.identity_stratum_key(dp, k, cc)`, `m_calib.stratum_for`). Test: `test_identity_rings_are_per_cadence_class` |
| (eval) cusum / mcusum acc_alarm set on the first live ticks of idle entities (pack B: every control, 00:15) | B14 restarts the charts at the end of warm-up / release / rebase, but `run['alarm']` (re-emitted by `_idle_outputs`) kept the warm-up latch | `_reset_charts` clears the cusum / mcusum alarm (rebase also bocpd / creep). Test: `test_restart_at_end_of_warmup_clears_the_reported_alarm_of_an_idle_entity` |
| (eval) stale `class:<id>` behavior.common.* series (robustness gate) | B05 skipped a group with no scored member, and a class with no active member | Written as undefined (L NaN, n 0, dir 0) every tick, for every class and the system. Test: `test_group_with_no_scored_member_is_written_as_undefined` |
| (eval) unknown_identity MEDIUM ('same class, different individual') on 13 of 19 enrolled humans in a clean week, with self posterior > 0.98 | B16's per-candidate typicality took chi2_r of the LDA distance, i.e. within-entity covariance I; B15's held-out genuine T99 is 200–1100 for workstations against chi2_14's 99 % point of 29 | The distance is calibrated by the candidate's T99 before the chi2 tail (`attribution.t99_scale`): p < 0.01 ⟺ d² > T99. Test: `test_typicality_is_calibrated_by_the_candidates_heldout_t99` |
| (eval) stale behavior.wh on the L6 / L8 newcomers | B06 wrote nothing on an observed tick once its density had been reset | NaN wh on an unscorable tick of a previously scored entity (a never-scored entity still writes nothing). Test: `test_reset_density_writes_undefined_wh_instead_of_going_silent` |
| (eval harness) threats that reopened an FP incident closed hours before their onset were scored as missed | B27 reuses the id on a reopen within 24 h, so `opened` stayed at the first opening | `metrics.split_episodes`: each (re)opening is an episode for detection / TTD / notifications; FAR and incidents-per-episode still count an incident id once (`far.n_reopened` reports the reopenings). Test: `test_reopened_incident_is_a_new_opening_for_detection_but_one_far_incident` |

Fixes the interrupted integrator had already landed (kept and verified, tests
in `tests/test_integration_shared.py`):
- B28 never rolls back into warm-up rows (`rollback_to` is floored at the last training tick).
- B14's charts use only zr that the entity's own baseline identifies (`m_baseline.own_support`), and restart at 0 on the first live tick.
- B26 clears warm-up evidence on the first live tick.
- B24 and B25 detect a bare version change through the gate.
- The B23 governor label queue.
- The `m_class` index.
- `m_baseline` quantiles.
- `store.timeline` risk from the vec ring.
- The contract retention table.

Verified consistent, with no change needed:
- `act.stream` is decoded only through `m_template.stream_rows` / `stream_ticks` / `stream_window` and `m_seq.stream_symbols`, and reweighted by `stream_frac`.
- The `behavior.common.<g>` keys that B05 writes are the ones that B18 and B25 read.
- The regime dict and the event `extra.state` from B28 are what B27 reads.

## 5. End to end, strict mode

`runner.run_pack('mini', seed=0)` and `run_pack('smoke', seed=0)` with the full
registry and `ctx.config['strict'] = True`:

| Pack | Ticks | Exceptions | Stale lib-3 series | Incidents |
|---|---|---|---|---|
| mini | 72 × 3600 s warm-up + 128 × 900 s | 0 | 0 | 9 |
| smoke | 120 × 900 s warm-up + 60 × 900 s | 0 | 0 | 17 |

Every real entity has finite `feature.vec`, `behavior.score`, `behavior.p`,
`e_day`, and `risk` on every live tick. The only all-NaN collected columns are
the following, and both are correct:
- `active` of class pseudo-entities: classes have no `feature.active`.
- `p` / `e_day` of a dissolved class (`erp-prod|class:r1` once B02 left it
  with one member): B26 and B28 keep writing its decaying risk and trust
  so that an open class incident can still close.

`tests/test_pipeline_e2e.py` runs the mini pack in strict mode and asserts:
- the registry is the full v2 set in contract order;
- 0 exceptions, 0 engine errors, 0 stale series;
- non-empty feature, score, p, e_day and risk series for all 8 entities;
- the baseline keeps n_eff > 0;
- an incident opens on a malicious-scenario entity and receives
  alarm / finding / risk evidence inside that scenario's own window;
- per-engine timings are recorded.

## 6. Cost per tick

Measured with `run_pack('smoke', seed=0)` (20 personas; about 12 active
during the weekend warm-up, 20 on the live Monday; Δt = 900 s; one core;
single-threaded BLAS). Budget = architecture §7 at 40 entities, halved for 20.

| Engine | Warm-up mean (ms) | Live mean (ms) | Live p95 (ms) | §7 budget @20 (ms) |
|---|---|---|---|---|
| raw (8 engines) | 5.0 | 7.5 | | – |
| derived.aggregation | 12.31 | 17.79 | 25.82 | – |
| derived.periodicity | 6.16 | 8.24 | 12.32 | – |
| derived.trend | 8.20 | 10.96 | 17.99 | – |
| derived ratio + entropy + graph + session | 3.49 | 4.97 | | – |
| B01 feature_vector | 5.05 | 6.52 | 10.06 | 6.00 |
| B02 peer_group | 4.41 | 1.28 | 10.17 | 0.50 |
| B03 baseline | 6.91 | 6.61 | 15.43 | 3.50 |
| B04 likelihood | 27.91 | 37.57 | 57.86 | 3.00 |
| B05 common_mode | 2.45 | 2.95 | 4.75 | 0.50 |
| B06 multivariate | 11.78 | 7.85 | 19.27 | 2.50 |
| B07 rhythm | 5.42 | 9.17 | 13.36 | 0.50 |
| B08 novelty | 6.39 | 9.70 | 17.75 | 1.50 |
| B09 client_identity | 2.66 | 3.85 | 6.75 | 0.50 |
| B10 sequence | 10.53 | 12.76 | 22.51 | 2.50 |
| B11 timing | 6.15 | 7.28 | 11.67 | 1.00 |
| B12 beacon | 0.61 | 0.96 | 1.97 | 1.00 |
| B13 budget | 7.07 | 9.80 | 14.81 | 1.00 |
| B14 changepoint | 8.45 | 8.93 | 15.08 | 1.50 |
| B15 identity_model | 7.66 (was 15.2) | 3.02 | 5.95 | 2.50 |
| B16 attribution | 13.23 | 30.60 | 52.20 | 5.00 |
| B17 entity_link | 1.93 | 4.22 | 16.49 | 0.50 |
| B18 class_monitor | 2.94 | 6.38 | 9.94 | 1.50 |
| B23 feedback | 0.06 | 2.47 | 4.69 | 0.75 |
| B24 calibration | 6.95 | 9.67 | 14.41 | 0.75 |
| B25 fusion | 4.08 | 5.64 | 8.81 | 0.75 |
| B26 risk | 1.82 | 2.59 | 4.09 | 0.75 |
| B27 incident | 0.22 | 2.60 | 5.21 | 0.75 |
| B28 governor | 2.42 | 4.64 | 7.11 | 0.75 |
| B30 portrait | 7.06 | 9.35 | 19.29 (was 65) | 2.50 |
| signature.rule_match | 3.04 (was 13.0) | 3.83 | 6.44 | – |
| signature.correlation | 0.65 | 0.91 | 1.48 | – |
| **lib-3 (behavior) total** | **154.2** | **206.4** | | **42.0** |
| **pipeline** | 193.3 | 261.0 | 386.1 | |

- The lib-3 warm-up totals 18.5 s over 120 ticks. The §7 smoke gate is ≤ 12 s for 180 ticks, so lib-3 is about 2.3× over that gate and 3.7× over the per-tick line.
- Changes in this pass: rule_match 13 → 3 ms (named snapshot), B15 warm-up 15 → 7.7 ms (calibration stride), B30 live p95 65 → 19 ms (per-key phase).
- Run-to-run noise on this host is about ±15 %. A second run gave a lib-3 total of 180 ms warm-up and 227 ms live.
- Mini pack (8 entities, 72 × 3600 s + 128 × 900 s): wall 32–36 s, pipeline 186 ms per warm-up tick and 125 ms per live tick.

## 7. Open issues (for W6 tuning and the engine owners)

1. **Cadence switch and distinct counts.** `distinct_templates` and
   `distinct_peers` are `count` features, so they are per-minute rates. A
   distinct count is sub-additive, so the rate at 900 s is about 4× the rate
   at 3600 s. After the mini pack's 3600 → 900 s switch every entity scores
   z ≈ +2 on them, and B06's T² on near-constant columns turns that into
   p ~ 1e-20 on the first live ticks. This is a FEATURE_SPEC-level question:
   a distinct-count kind with a cadence-aware normaliser. It also affects
   B15 and B16 windows (IQR and median of per-minute rates across a cadence
   switch).
2. **B06 SPE on machine personas.** For x.x.9.9 (sanctioned automation, L15),
   SPE pm sits at 1e-7 to 1e-17 through the warm-up. The out-of-sample
   Box calibration with n ≈ 100–300 rows in 52 dimensions does not hold.
   B24 blends pm while a stratum holds fewer than 64 entries, so these
   entities alarm on the first live tick.
3. **Weekend-only warm-ups.** The smoke pack warms up Sat 18:00 → Mon 00:00,
   so workday buckets are empty, and human workstations first appear on
   Monday. They are scored against a machine-dominated system tier: marg_int
   z ≈ −5, `new_entity_unmatched` and `unknown_identity`. Either the pack or
   the cold-start backoff (a class prior for unmatched newcomers) needs a
   decision.
4. **The runtime's 900 → 60 s switch.** This is `Runtime` and
   `scripts/smoke.py`. Live strata start empty, and B24 falls back to pm
   for the first 64 entries. The routine-lib-4 part of the risk flood is
   fixed (§4). Item 1 is much stronger here, though: a 4-minute B16 window
   of per-minute distinct rates is far from the 1-hour windows B15
   enrolled. Within ~10 live minutes every enrolled entity gets
   `unknown_identity` ("fits its class but no known individual") and an
   incident. The smoke script shows 16 open incidents after 16 live ticks.
   This is the most visible open issue for the product. The fix belongs to
   the FEATURE_SPEC (item 1) plus a cadence-aware B15 enrolment (windows by
   wall-clock span, not by K ticks).
5. **Perf budget.** B04, B16, B10, B14, B07 and B08 are 3–15× over the §7
   line even after the strides above. The exact predictives (B04: 3 × 52
   exact NB / BB / t mid-p per entity-tick through scalar paths, ~2.5 ms per
   entity) and PPM scoring (B10, B16) are the floor of the current design.
   Getting further needs vectorised BB and NB mid-p kernels across features,
   or a compiled PPM. The B18 author made the same point about the §7 line
   (R17.3). Update: §9 has the exact (results-unchanged) performance pass. It
   cut about 20 % of pack A / B wall time and brought memory under the gate,
   and it lists what is still needed for gate 14.
6. **Deferred requests.** R0.2, R6.1, R17.1, R20.3 and R21.3 are optional or
   owner-side. The B29 parts of R5.3 and R15.3 wait for B29.

## 8. First evaluation run (seed 0, strict)

Packs mini, smoke, A and B, seed 0, `ctx.config['strict'] = True`, run through
`runner.run_pack` + `metrics.score_run` + `report.write_report`. Reports:
`reports/before/`, `reports/` (after the fixes in §4 marked "(eval)") and
`reports/experiment_A_900s_warmup/` (pack A with its 336 × 3600 s + 192 × 900 s
warm-up replaced by 672 × 900 s; an experiment, not a gate pack). Wall time:
mini 30 s, smoke 40 s, A 850 s, B 1460 s per pack-seed (gate 14: 360 s).

| | A before | A after | B before | B after | A, 900-s warm-up (after) |
|---|---|---|---|---|---|
| threats detected / within deadline | 1/12, 1/12 | 1/12, 1/12 | 1/10, 1/10 | 1/10, 1/10 | 7/12, 5/12 |
| FAR ≥ LOW / ≥ MEDIUM per entity-day | 0.25 / 0.25 | 0.25 / 0.197 | 0.083 / 0.076 | 0.083 / 0.058 | 0.37 / 0.22 |
| HIGH / CRITICAL control incidents | 19 / 9 | 10 / 6 | 16 / 3 | 10 / 5 | 5 / 1 |
| single-tick e_day ≤ 0.03 on clean control ticks (expected) | 1309 (2.3) | 1142 (2.3) | 1374 (8.3) | 1416 (8.3) | 42 (2.3) |
| evidence-CUSUM alarms | 102 | 49 | 9 | 1 | 16 |
| stale lib-3 series | 6 | 0 | 15 | 0 | 0 |

**The cadence switch dominates packs A and B.** Every pack warms up 14 d at
3600 s (+ 2 d at 900 s) and scores at 900 s. Features that are not
exposure-exact change their null with the tick length: distinct counts
(`distinct_peers`, `distinct_templates`, `distinct_dports`, `ja3_diversity`:
a distinct count is sub-additive, so its per-minute rate is ~4x higher at
900 s), log-rates and log-averages of bursty quantities (`bytes_*`,
`bytes_per_flow`, `updown_log`: Jensen), K-tick identity windows (4 h vs 1 h)
and 6-h window statistics (`timing_regularity`). After 12 live days of pack B
the control entities still have zr ≈ +1.5 to +2.3 on `distinct_peers` /
`distinct_templates` (the reference anchor moves ≤ 0.03 σ15 per day), so the
k = 0.25 CUSUM / MCUSUM bank is latched on ~80 % of control ticks, every
control entity has an incident from the first live hours, incidents never
close, and the threats that come later only escalate incidents that opened
before them (a TP must open inside the scenario window). With the same code
and a 900-s warm-up (last column) calibration is within ~20x of nominal
instead of ~500x, and 7 of 12 threats open their own incident. Fixing it is a
FEATURE_SPEC / B03 decision (open issue 1): e.g. distinct counts over a fixed
trailing wall-clock window (union of the per-tick sets over 1 h), baselines
keyed by cadence class for the non-exposure-exact families, or packs that warm
up at the scenario cadence.

Tuning / design candidates found in the run (not changed):
- B24's small-sample prior is `behavior.pm` when the entity's own rings are
  short. For a sparse entity (a nightly backup host: 3–4 scored ticks a day)
  the peer pm is 1e-300 every night, so its p is at the floor every night; for
  accumulators p_eq = 1 at a zero statistic reinstates the point mass at p = 1
  (creep KS D = 1.0, budget_* conservative).
- B25 meta rings admit warm-up ticks with trust 1 by definition. Warm-up
  extremes (p_all 1e-21 … 1e-37 from the cases above) make the meta tail heavy
  (xi ≈ 0.3), after which q_all saturates near 5e-4: T17 on 10.40.4.53 has
  family p 1e-37 and e_day 0.05 (no single-tick alarm). A release commits held
  rows with trust_prov, which ignores accumulator evidence, so a breadth
  accumulator at p 1e-28 entered a live meta_all ring.
- B07 `machine_like` (normalised 168-bin entropy ≤ 0.8) is true for most office
  workers.
- B28 REJECTs (and freezes) on the first live tick when lib-4 ≥ HIGH evidence
  meets a rhythm-type suspect (the prior is 0 and lib-4 adds −2.3).
- B16: with the T99-calibrated typicality, the T9b intruder no longer raises
  unknown_identity (humans' own held-out windows reach d² 200–1100); identity
  needs the cadence-aware windows of open issue 4 before its power can be
  judged.
- Perf (gate 14): lib-3 p95 ≈ 1.3 s per tick scaled to 35 entities.

### 8.1 spec v2.1: canonical grain mode (docs/lib3/cadence.md)

Pack A seed 0, strict, `grain_mode='canonical'` (the default since M8) and
the §10 warm-up (312 × 3600 s + 288 × 900 s, Fri–Sun at the live cadence),
through `scripts/evaluate.py` (runner → metrics → report). Scenario phase,
seeds and onsets unchanged. "A after" is the column above.

| | A after (v2, tick) | A, spec v2.1 |
|---|---|---|
| threats detected / within deadline | 1/12, 1/12 | 12/12, 6/12 |
| FAR ≥ LOW / ≥ MEDIUM per entity-day | 0.25 / 0.197 | 0.30 / 0.197 |
| HIGH / CRITICAL control incidents | 10 / 6 | 2 / 1 |
| single-tick e_day ≤ 0.03 on clean control ticks (expected) | 1142 (2.3) | 12 (2.3); H ticks 2 (1.5), Q ticks 10 (0.8) |
| evidence-CUSUM alarms | 49 | 14 |
| B14 "change"-path accumulator onsets | — | 11 |
| worst detector KS D (n ≥ 100) | — | 0.42 (spe) |
| portrait p5–p95 coverage: ticks / H rows / Q rows | — | 0.98 / 0.96 / 0.97 |
| stale lib-3 series / exceptions | 0 / 0 | 0 / 0 |
| wall time (ticks) | 850 s (912) | 939 s (984) |

The cadence switch no longer dominates: the single-tick rate on clean control
ticks is within ~5x of nominal (it was ~500x), the CUSUM bank no longer
latches on the distinct-count bias, and every threat opens its own incident.
The FAR ≥ LOW rate is now spread over the control population (about one incident per
entity, mostly MEDIUM, axes shape / volume / identity) instead of a few
entities latched from the first live hour. Two defects were found and fixed on
the way (cadence.md §17): the H-stream evidence reset at the first live tick
was lost when that tick was not an H tick, and B14 did not report a latched
alarm between H ticks. Open (cadence.md §17): B11 timing p on warm-ups whose
live-cadence days are a weekend (B24's pm prior), and B06 densities that keep
zi rows scored before the entity's own baseline existed (short warm-ups:
mini / smoke).

`scripts/smoke.py --strict` (Runtime plan 120 × 3600 s + 192 × 900 s, 16 live
ticks at 60 s): SMOKE OK; 10 open incidents (v2: 16), 5 of them on the 5 demo
threat entities, the other 5 LOW except one MEDIUM (api-gateway/10.40.9.9).
The Runtime warm-up ends at the wall clock, so the day types its 900-s phase
covers depend on the weekday it is run (it warns when one is missing).

### 8.2 W7 tuning fixes (root causes of §8's design candidates)

Each fix is at its root cause, with a regression test; no constant was set
from a pack-seed outcome. Pack A seed 0 was used to measure, with every fix
switchable by monkeypatch (scratch driver), so each row below is an
ablation on the same tree; "baseline" (all fixes off) reproduces §8.1
exactly.

| Candidate (§8) | Root cause | Fix | Tests |
|---|---|---|---|
| B24: a sparse entity's p at the floor every night; creep KS D = 1.0, budget_* conservative | The small-sample prior was behavior.pm, a p-value with two atoms: an accumulator's stationary p_eq is exactly 1 at a zero statistic, and a sparse nightly host scored against its peers had pm at the float floor on every null night. As a logit prior an atom puts the issued p back in a point mass at 1, or at the floor | `m_calib.pm_prior`: a tie at an atom is randomised with the mid-p rule, pm = 1 → 1 − π1 + u·π1, pm at the floor → u·π0 (u the score's seeded U), π the atom's share of the entity's admitted pm history (ring `<d>@pm\|<cc>`, canonical H / Q `<d>@pm\|g:<g>`; below 16 entries π1 Laplace, an unseen floor kept). The body of pm is used as is. B29 replays it (`p_replay`) | `test_b24_b25_tuning.py` (floor, atom, rules); tick-mode golden |
| B25: warm-up extremes in the meta rings (ξ ≈ 0.3, q_all saturating near 5e-4; T17 family p 1e-37 → e_day 0.05); a release admitted a breadth accumulator at p 1e-28 | −log10 of a valid p has an exponential tail, so a heavy fitted tail is contamination; releases commit with trust_prov, which has no accumulator factor | Tail: exceedances beyond the 99.9 % bound of the maximum of n_u exponential excesses (rank-based scale) are replaced by their expected order statistics before the PWM fit (`calib.winsorise_exceedances`, `fusion.META_WINSOR_ALPHA`; a clean fit is touched in ~1 % of refits). Release: B28 writes `behavior.trust_evidence` on live ticks (no alarm, no finding ≥ MEDIUM, every accumulator < h/2) and B24 / B25 admit with min(weight, `m_governor.evidence_weight`), which binds for released rows | `test_b24_b25_tuning.py` (winsorised tail, null touch rate, release caps, warm-up not gated), `test_b28_governor_paths.py` |
| B07: `machine_like` true for most office workers | Normalised 168-bin entropy is not monotone in automation: office worker ~0.75 (≤ 0.8), 24/7 client ~1.0, nightly job ~0.45 | Automation index `m_rhythm.automation_index` = mean of the off-hours activity ratio, the non-workday / workday ratio, presence regularity 1 − Σr·4r(1−r)/Σr, and B02's per-IP A; B02's hysteresis (0.6 / 0.4); never machine-like when regularity < 0.5 (silence and a missed window mean nothing for coin-flip presence) | `test_b07_automation.py`; `test_b07_rhythm.py::test_b/test_c` (assertions moved from entropy to the index); slow `tests/test_b07_automation_generator.py` |
| B28: REJECT + freeze on the first live tick (rhythm suspect + lib-4 HIGH: prior 0 − 2.3 → P = 0.09) | REJECT had no persistence requirement, unlike ACCEPT (T_type) and SUSPECT → DRIFTING (1 h, 4 ticks) | `m_governor.reject_corroborated`: a REJECT needs the episode's own alarm / accumulator evidence on ≥ 4 ticks spanning ≥ 1 h, or two independent malicious sources, or a tp label; otherwise it holds and returns once quiet | `test_b28_governor_paths.py` (3 new) |
| W7: profile.extra.feedback.n_labelled 0 after a label | B23 wrote the profile summary inside `_apply_label`, before `_relearn` updated n_labelled | Profiles written after the refit | `test_b23_feedback.py::test_profile_feedback_summary_counts_the_label_just_applied`, API round trip |
| W7: behavior.degraded never written | Only NaN-producing failures wrote it | `emit.cause` / `emit.write_degraded`, kinds stale / producer_error / unscorable (NaN) and insufficient_support / fallback / provisional (still scored). B07 offhours judged mostly by its tier prior (`fallback:<tier>`), B18 class_int / class_shape before the reference anchor holds data, B24 for Q scores on the provisional transfer. `/api/detectors/health` reports the degraded share over the keys running each detector, with cause kinds; the eval stale checker treats behavior.degraded as sparse by contract | `tests/api` |
| W7: portrait internals used by the API | `routes_v2` imported `portrait._signature` / `_safe_token` | `lib/m_portrait` (signature, diff_signatures, safe_token, …), re-exported by portrait.py; `views.portrait_diff` / `views.mask_template` | `tests/api` |
| W7: narrative | `views.narrative` labelled the zh fallback as en when only zh existed | reads narrative_zh / narrative_en / headline_* from B29's explanation; Incident.narrative only when neither language exists | `tests/api` |

Pack A seed 0 (strict, canonical; single-tick exceedance = realised /
expected count of e_day ≤ 0.03 on clean control ticks, expected 2.28, H
1.52, Q 0.76; CUSUM = evidence-CUSUM alarms per control entity-day):

| Variant | exceed ≤ 0.03 all / H / Q | ≤ 0.003 | CUSUM | FAR ≥ LOW / ≥ MED | HIGH+ / CRIT |
|---|---|---|---|---|---|
| baseline (all off) = §8.1 | 5.3 / 1.3 / 13.2 | 43.9 | 0.18 | 0.303 / 0.197 | 3 / 1 |
| **final** | 8.8 / 0.7 / 25.0 | 39.5 | 0.28 | 0.303 / 0.224 | 3 / 0 |
| final without the release cap | 5.3 / 1.3 / 13.2 | 21.9 | 0.25 | 0.303 / 0.184 | 2 / 0 |
| rejected: warm-up rows also gated by their own evidence | 18.4 / 7.2 / 40.8 | 74.6 | 0.28 | 0.289 / 0.224 | 3 / 0 |
| rejected: … and by the q_inst evidence factor | 61.8 / 75.7 / 34.2 | 342 | 2.45 | 0.316 / 0.224 | 6 / 2 |
| rejected: whole of pm calibrated on its ring (B25 fixes off) | 34.2 / 44.1 / 14.5 | 219 | 1.66 | 0.303 / 0.211 | 1 / 1 |

- Per detector (KS D on trusted ticks, baseline → final): creep 0.227 →
  0.046, novelty 0.058 → 0.009, silence 0.019 → 0.009, offhours 0.045 →
  0.033, budget_vol 0.078 → 0.053; spe / identity / mcusum / bocpd / t2 /
  jsd stay 0.24 – 0.41 (their live score distribution differs from their
  warm-up rings: B04 / B06 / B14 / B16). Accumulator-path alarm rates per
  control entity-day: budget 0.039 → 0.013, change 0.145 → 0.118,
  temporal_categorical 0.066 → 0.039. Threat detection unchanged (12
  scenarios, same within-deadline set); T12 TTD 9 → 5 ticks.
- Two designs were measured and rejected. (1) Gating the meta rings' warm-up
  rows by their own evidence, as live learning does: any gate that depends
  on the row's evidence (the q_inst factor is the rings' own output; even
  accumulator ≥ h/2 alone) truncates the null tail the ring estimates, and
  every threshold then fires too often live. (2) Calibrating the whole of pm
  on the pm ring: per detector it was uniform, but the meta rings had been
  absorbing the live miscalibration of spe / identity / mcusum / bocpd
  (KS 0.24 – 0.41) through the extreme warm-up pm; with those gone the
  fused single-tick rate rose 34 – 60x. The atoms-only rule keeps what the
  candidate named (the floor and p_eq = 1 point masses).
- The release cap is kept for consistency with the live gate (a normal
  live commit's trust already excludes rows with an accumulator ≥ h/2);
  on this seed it costs 8 single-tick exceedances (Q ticks 10 → 19) and
  3 MEDIUM incidents over 76 control entity-days, the truncation effect
  above on the few released rows. Worth re-measuring on seeds 1 – 4.

B07 on the generator population (37 base personas, 10 clean warm-up days at
3600 s, seeds 0 and 1; reproduced by the slow test): every human persona
(interactive, search, NAT; 36 entity-runs) indexes 0.14 – 0.33, every
machine persona (API, integration, health, backup; 38) 0.73 – 0.93. The v2
rule called 22 of the 36 human runs and 4 of the 38 machine runs
machine-like (it missed every 24/7 client).

Tick-mode golden (`tests/test_tick_mode_golden.py`): with these fixes
switched off by monkeypatch the fingerprint matched the recorded one
exactly, so its difference is only the deliberate changes above, and it
was regenerated. Of the 4719 issued p in its 48 live ticks, 1632 were
exactly 1 before (the p_eq = 1 point masses of the small-sample blend) and
53 after; 1579 p changed; alarm paths / severities and the 8 incidents are
unchanged.

`scripts/smoke.py --strict` (Runtime plan 120 × 3600 s + 192 × 900 s, 16
live ticks at 60 s). Its warm-up ends at the wall clock, so runs differ by
the day they are made; compare runs made together.
- Sunday run, before vs after (an intermediate B25 variant): 12 → 8 open
  incidents; the backup host 10.20.9.5 temporal incident, two LOW "change"
  incidents and a LOW categorical on 10.40.4.51 were gone.
- Monday 09:13 (Asia/Shanghai) run, baseline vs final side by side: 12
  incidents each, on the same 12 keys (5 demo threats, 7 clean). Both are
  led by class:r3 (humans) temporal HIGH on two systems at the
  Monday-morning start after a Fri–Sun 900-s phase (B18 class_rhythm), not
  by these engines.
- The remaining post-switch risk on clean keys is the two health checkers
  (10.40.9.9 / 10.30.9.9, risk 64 – 72, their class keys with them): 78 %
  of it is the lib-4 signature `c2_beacon` (HIGH, `data/signatures/
  primitives.yaml`), whose clause `http.requests ≤ 10` is a per-tick count.
  A 30-s poller makes 30 requests per 900-s tick (no match in the whole
  warm-up) and 2 per 60-s tick (a match on every live tick), so the rule is
  cadence-dependent; B08 then scores the lib-4 category token 'beacon' as
  first_seen at system tier (the other 20 %), and B26 counts HIGH lib-4
  matches without habituation. The fix belongs to the signature rule
  (express the volume clause as a rate) — not in B24 / B25 / B28.

## 9. Performance pass (results unchanged)

Scope: lib-3 CPU and memory measured after the cadence refactor (§8.1),
without changing any result. The method was:

- cProfile of the whole of pack A (seed 0) and of the smoke pack.
- The runner's per-engine timings.
- A side-by-side harness: the pre-pass tree and the same tree plus the
  changed files, run on the same host at the same time.

Every change below is exact. The checks were:

- **mini pack:** every vector-ring write, derived point, event, incident,
  model object, profile and stale-series record is identical at relative
  tolerance 0.
- **packs A and B (20 and 29 simulated days):** events, incidents (with
  history and explanations), the collected per-tick series, portraits and
  the held-out snapshots are identical at tolerance 0.
- **Unit equivalence tests:** each rewritten routine is compared with its
  reference (old) implementation on random inputs:
  - `tests/engines/test_b04_batched.py`
  - `tests/lib/test_perf_equivalence.py`
  - `tests/core/test_store_perf_paths.py`
  - `tests/eval/test_runner_perf.py`
  - `tests/engines/test_identity_bg_memo.py`
  - `tests/engines/test_b10_tier_reuse.py`
  - `tests/core/test_gc_config.py`

### 9.1 Changes

| Where | What | Why it is exact |
|---|---|---|
| B04 likelihood | One batch per tick. The engine collects the rows it scores (H and Q, every entity). It builds all predictives at once (`m_baseline.predictive_set_many` / `predictive_q_many`: the per-entity chain / join / leave-one-out assembly is unchanged, and `_params`, `_refine_par`, `_ebar`, `_mean` run once on the stacked rows). The NB mid-p of every row, anchor and count feature is one call to the array kernel. The model_state quantiles of every due row are one `m_baseline.quantiles_many`, and the BB quantile groups of a call share one zero-padded pmf pass (`bayes._bb_ppf_small_many`). The PIT seed prefix is rendered once per row. | Parameter maps and quantile searches are element-wise per row. The array NB kernel is bit-identical to the scalar one. A row cumsum over zero padding is the same sequential sum. The BB mid-p / pmf, the NB pmf and the observation transform stay on their scalar per-element paths (see 9.3). |
| m_identity (B15, B16, B17) | `Background.term` computes the candidate-independent background side of every modality LLR (system seq loglik, vocab, client, rhythm, timing) once per (data, tick) instead of once per candidate. | The same computation. The memo holds the data object, so its id is never reused within a tick. |
| m_seq.top_ngrams (B10 describe, B30 portraits) | A stable descending argsort of the scaled counts. Only the k winners are turned back into n-grams. | `heapq.nlargest` is `sorted(..., reverse=True)[:k]`, which is stable, so ties keep dict order in both. The values are the same products. |
| ppm.merge | An order-0 target (the B10 system tier) reads only the empty context instead of skipping every longer one. | The skipped contexts were `continue`d before. |
| B10 tiers | `_build_tier` republishes the previous tier (a new version at now) when no member's learned state changed since this key's last build. A per-entity marker (state identity, journal head and length, last_ts, version, branch, held, link, frozen, rollback and applied directives) moves on every gated update. | A tier is a pure function of its members' states in member order. `test_b10_tier_reuse` compares against an engine that always rebuilds, tick by tick. |
| store | `vec_at` answers the newest row, or a ts past it, without a search. An unwrapped ring is searched in one `searchsorted`. `drop_before` returns at once when the oldest row is still inside the retention. The pseudo-entity test is memoised. There is a new `names_signature` (contract.md store API). | The same rows are returned or kept (`test_store_perf_paths`, including wrapped rings). |
| eval runner | The stale checker's O(derived × vector names) scan per entity and tick is cached on `store.names_signature`. It took 91 of the 1775 profiled seconds of pack A. | The name sets only grow, so equal sizes mean equal sets (`test_runner_perf`). |
| pipeline (orchestrator) | `configure_gc` raises the cyclic-GC thresholds to (10000, 20, 50) when the process kept the interpreter defaults (700, 10, 10). The store and the models form a large heap of mostly long-lived objects (about 0.9 M tracked objects on pack A). At the defaults a full collection ran every few ticks: 56 full collections and 31 s of GC in the first 400 s of pack A, with 0.3–1.3 s pauses that were charged to whichever engine was running at the time (the B08 "spikes" at the live p95). With the new thresholds the same window has 0 full collections and 8 s of GC. | The collector only decides when unreachable cycles are freed, and almost everything here is freed by reference counting. Every id()-keyed cache in lib-3 also keys on a version or holds its object. Pack A is identical. |
| retention audit (contract B table) | `behavior.acc_alarm`, `rhythm`, `timing`, `id`, `class`, `behavior.common.*` (not `common.q.*`), `behavior.cp.*` and `behavior.seq.class_llr` had no rule, so they kept up to 20000 points (208 d at 900 s). They now keep 8 d. | Every reader looks at most a few points back. The deepest is B10's gated-learner clock `seq.class_llr`, with 192 h of replay like `feature.nat`. Packs A (20 d) and B (29 d) are identical. The API history of these series is now bounded at 8 d, like the other lib-3 long rings. |

### 9.2 Before / after

Pack A, seed 0, strict (984 ticks: 312 × 3600 s + 288 × 900 s of warm-up
and 384 live × 900 s; 43 entities; one core; single-threaded BLAS). The
pre-pass tree and the final tree ran at the same time on the same host, and
their outputs are identical. Mean ms per tick by phase; the live p95 is per
engine.

| Engine | warm-up 3600 s (ms) | warm-up 900 s (ms) | live 900 s (ms) | live p95 (ms) | pack total (s) |
|---|---|---|---|---|---|
| B10 sequence | 170.0 → 120.6 | 110.7 → 65.0 | 125.5 → 98.8 | 198.6 → 231.1 | 133.1 → 94.3 |
| B30 portrait | 184.8 → 160.2 | 78.4 → 53.6 | 85.1 → 54.9 | 144.6 → 103.9 | 112.9 → 86.5 |
| B04 likelihood | 99.8 → 42.8 | 80.3 → 38.3 | 121.1 → 54.7 | 292.2 → 128.3 | 100.8 → 45.4 |
| B13 budget | 48.6 → 47.9 | 38.9 → 37.8 | 62.0 → 58.9 | 102.3 → 90.5 | 50.2 → 48.5 |
| B06 multivariate | 71.4 → 69.1 | 34.0 → 33.1 | 43.4 → 41.4 | 113.2 → 111.6 | 48.7 → 47.0 |
| B16 attribution | 60.6 → 54.6 | 18.0 → 16.7 | 29.4 → 27.2 | 159.2 → 148.6 | 35.4 → 32.3 |
| B01 feature_vector | 26.9 → 25.6 | 32.7 → 32.5 | 41.8 → 39.7 | 68.0 → 59.5 | 33.8 → 32.6 |
| B24 calibration | 30.8 → 30.1 | 31.4 → 32.4 | 33.1 → 30.8 | 59.8 → 55.5 | 31.4 → 30.5 |
| B03 baseline | 35.6 → 34.2 | 24.8 → 24.4 | 28.5 → 27.4 | 59.7 → 55.8 | 29.2 → 28.2 |
| B08 novelty | 21.7 → 20.2 | 17.5 → 15.1 | 41.8 → 28.8 | 61.1 → 48.6 | 27.9 → 21.7 |
| B29 explain | 0.0 → 0.0 | 0.0 → 0.0 | 62.1 → 56.0 | 282.3 → 249.3 | 23.9 → 21.5 |
| B07 rhythm | 28.3 → 23.5 | 19.1 → 16.2 | 24.4 → 17.6 | 33.0 → 25.9 | 23.7 → 18.7 |
| B25 fusion | 16.2 → 14.8 | 16.0 → 15.6 | 23.0 → 18.0 | 31.5 → 27.9 | 18.5 → 16.0 |
| B15 identity_model | 35.3 → 32.0 | 12.2 → 11.3 | 10.1 → 7.6 | 33.2 → 26.5 | 18.4 → 16.1 |
| B14 changepoint | 31.2 → 30.4 | 9.2 → 9.0 | 11.1 → 10.6 | 51.8 → 46.3 | 16.6 → 16.1 |
| B11 timing | 15.4 → 15.3 | 11.5 → 11.4 | 17.0 → 16.3 | 26.3 → 26.2 | 14.7 → 14.3 |
| B18 class_monitor | 16.6 → 16.4 | 10.4 → 10.2 | 12.2 → 11.5 | 24.2 → 22.2 | 12.9 → 12.5 |
| B28 governor | 8.7 → 7.9 | 10.2 → 9.5 | 16.9 → 14.6 | 28.7 → 23.2 | 12.2 → 10.8 |
| B09 client_identity | 7.8 → 7.6 | 5.8 → 5.6 | 9.3 → 9.0 | 13.9 → 14.3 | 7.6 → 7.4 |
| B26 risk | 5.9 → 5.6 | 6.8 → 6.7 | 9.1 → 8.2 | 14.8 → 11.9 | 7.3 → 6.8 |
| B05 common_mode | 4.9 → 4.4 | 6.7 → 5.7 | 9.2 → 7.2 | 22.1 → 13.1 | 7.0 → 5.8 |
| B17 entity_link | 9.7 → 9.2 | 2.8 → 2.7 | 4.8 → 4.5 | 30.6 → 25.1 | 5.7 → 5.4 |
| B02 peer_group | 7.4 → 6.9 | 3.4 → 3.4 | 4.1 → 3.4 | 32.4 → 28.9 | 4.9 → 4.4 |
| B12 beacon | 1.3 → 1.2 | 1.2 → 1.1 | 8.4 → 7.7 | 15.0 → 12.1 | 3.9 → 3.6 |
| B27 incident | 0.1 → 0.1 | 0.1 → 0.2 | 4.3 → 3.9 | 8.2 → 6.7 | 1.7 → 1.6 |
| B23 feedback | 0.1 → 0.1 | 0.1 → 0.1 | 3.7 → 3.6 | 7.0 → 6.8 | 1.5 → 1.4 |
| **lib-3 total** | 939.2 → 780.7 | 581.9 → 457.4 | 841.5 → 662.0 | 1657.1 → 1117.0 | 783.8 → 629.5 |
| **pipeline** | 1017.7 → 854.6 | 652.0 → 526.6 | 971.2 → 777.0 | 1854.9 → 1242.3 | 878.2 → 716.6 |

| Pack A (seed 0) | before | after | gate |
|---|---|---|---|
| wall time (s) | 932 | 740 (−21 %) | 360 |
| lib-3 CPU (s) | 784 | 630 (−20 %) | — |
| eval-harness collect time (s) | 31.0 | 6.8 | — |
| live lib-3 p95 per tick (ms) | 1657 | 1117 | — |
| … scaled to 35 entities (ms) | 1349 | 909 (−33 %) | 80 |
| memory per entity at the end (MB, store.memory_report) | 10.7 | 9.4 | 12 |
| … projected to 8 live days (MB) | 13.4 | 11.7 (passes) | 12 |

Pack B, seed 0 (1824 ticks, 40 entities, 12 live days), same set-up;
outputs identical:

| Pack B (seed 0) | before | after | gate |
|---|---|---|---|
| wall time (s) | 1650 | 1315 (−20 %) | 360 |
| lib-3 CPU (s) | 1405 | 1126 | — |
| live lib-3 p95 scaled to 35 entities (ms) | 1381 | 919 | 80 |
| memory per entity at the end (MB) | 16.1 | 11.6 (passes) | 12 |
| … projected to 8 live days (MB) | 13.7 | 10.4 | 12 |

Pack B's end-of-run 16 MB was the unbounded dict series of the retention
audit: 12 live days of per-tick dicts, on top of the warm-up.

The smoke pack (20 entities; 96 × 3600 s + 24 × 900 s): two concurrent
base / final pairs.

| | before | after |
|---|---|---|
| lib-3 warm-up total (s) | 40.8 / 39.9 | 35.5 / 35.7 |
| lib-3 per tick, warm-up / live (ms) | 340 / 262, 333 / 259 | 296 / 233, 298 / 217 |
| B04 per tick, warm-up / live / live p95 (ms) | 44 / 54 / 121 | 24 / 27 / 55 |
| wall (s) | 69.9 / 68.6 | 59.8 / 59.2 |

B04 is about 2.3x faster. B10 is 1.4x faster in the warm-up, where its
class tiers were rebuilt from unchanged members every hour. The runner's
collection step is 4.5x faster. The GC change reaches every engine
(B07 / B25 / B28 are faster without being touched), and it removed the
largest p95 spikes. B30's gain comes from `m_seq.top_ngrams` and the GC
change; B30's own code was not touched.

### 9.3 Tried and not kept

- **A vectorised Beta-Binomial mid-p** (gammaln log-pmf, tail sums in a
  padded matrix, grouped by side length). It ran about 5x faster than the
  scalar path and agreed with it to about 1e-11. That is enough to flip a
  float32 ring value now and then: on pack A, B06's Q T² and SPE then
  drifted on one entity from scenario tick 210, and e_day, risk and one
  incident's explanation fidelity followed. The engine now uses the scalar
  BB path, which is bit-identical. The batched observe was dropped for the
  same reason: its CLR centring is a row mean, and numpy reduces a batch in
  a different order.
- **Faster `ppm.merge` loops** (dict comprehensions; C-level map / zip).
  Contexts hold 1–2 symbols on average, so the per-context overhead
  dominates. Both variants were 3–40 % slower on 25k-context models.
- **Precomputed context tuples in `ppm.loglik`.** No gain.
- **An early exit in `gating.commit_candidates` via `store.last_write_ts`.**
  It is not a store call the gating contract allows
  (`test_only_contract_store_calls`).

### 9.4 Where the time is now, and what is left

Where the time goes now (pack A totals):
- **B10, 94 s.** When a member has changed, a class tier is still rebuilt in
  full from every member's dict-of-dict PPM (about 1 µs per context and
  about 25k contexts per model). Checkpoint pickling is the other large
  part.
- **B30, 86 s.** `_workload_block`, `_grid_quantiles` and `_nb_cdf_rows`
  (owner: the portrait workstream).
- **B13, 49 s.** Round-robin tail and guard fits.
- **B06, 47 s.** Robust refits (C-step, OAS, EM).
- **B04, 45 s.** Predictive assembly and the scalar BB tails.
- **Smaller:** B01 (33 s), D0 aggregation (31 s), B16 (32 s) and B29
  (22 s, counterfactual replays, 0.25–0.4 s bursts).
- **The gated learners' per-step overhead** (commit candidates, checkpoint
  dumps, control): about 10 % spread over 12 learners.

Gate 14 still fails. The wall time is 740 s against 360 s, and the lib-3
p95 at 35 entities is 0.9 s against 80 ms. Every remaining cost is Python
work in the current algorithms, and none of it can be made several times
faster without changing a result. Each of the next steps needs a decision:
- **Compiled kernels.** PPM scoring and merging and the BB tail sums, in C
  or numba. This is a new dependency, and the summation order would have to
  be kept for bit-identity.
- **Incremental tier statistics** for B10 and B08. Members would add their
  committed rows to the class and system tiers at commit time instead of
  the hourly rebuild. The float summation order changes, so this is a spec
  change with a tolerance, not a refactor.
- **B30 and B29 off the tick path.** A portrait refresh on read, and a
  counterfactual explanation computed asynchronously per incident.
- **A per-tick CPU budget for B13 / B06 refits.** It would bound the
  per-tick share instead of the per-day share.

## 10. Evaluation rounds 2 and 3 (evaluator)

### 10.1 Runs and method

- **round-2 baseline** (`reports/round2_baseline/`): the tree after the perf
  (§9) and W7 tuning (§8.2) passes, before any evaluator fix. Packs A and B
  seeds 0 and 1, C / D / E seed 0; strict, canonical grain mode.
- **round 2** (an evaluator interrupted by a container restart; its fixes are
  in commit fdcde68 and listed in §10.2). "HEAD" below is that commit.
- **round 3** (this section): full suite, the lead's four items (Part B test
  module, `c2_beacon`, B18 class_rhythm after a short 900-s warm-up, the
  pack-A reopenings), diagnosis of every failing gate / missed threat /
  legit scenario over its severity down to the engine chain, fixes with
  regression tests (§10.3), final runs (§10.4). "final" is HEAD + §10.3.
- Every run: `runner.run_pack(pack, seed, strict=True)` + `metrics.score_run`,
  in parallel processes (scratch driver = `scripts/evaluate._job` + pickling
  of the RunResult for the chain diagnosis). The engine chain of an
  incident was read from the collected per-tick series (behavior.p of every
  detector, p_family, e_day, alarm path / severity, acc_alarm, risk) at its
  (re)opening and escalation ticks, the B26 risk components from the
  profiles and the events' extras.
- Compute: 4 cores. The interrupted evaluator's last job batch (25 jobs:
  full / feedback / ablation runs of the tree as of 03:54, i.e. HEAD minus
  its B13 count-scale floor) could not be stopped from this session and ran
  alongside, so every round-3 run shared the machine with 4 of its
  processes; gate 14 timings in this section are therefore not comparable
  with §9 (they are 1.5–2.5x slower). Its paired ablation / feedback runs are
  used for gate 13 (§10.6) with that caveat.

### 10.2 Round-2 fixes (committed in fdcde68)

| # | Where | Symptom (evidence) | Root cause | Fix | Test |
|---|---|---|---|---|---|
| 1 | generator | Pack E: every control entity CRITICAL at the 900 → 60 s switch; FAR ≥ LOW 1.0 | Human sessions were laid out inside each tick, so at 60 s every minute started a session (4x the login redirects and DNS lookups, 2.5x the POST share) | Sessions are a continuous-time process (Poisson starts, geometric sizes, events carried across ticks) | `test_generator.py` (2) |
| 2 | generator + R1 | `l4.retransmit_rate` of aggregated ticks ~w× the event-mode rate | the group's retransmit total was put in a per-flow field | `extra['retransmits_total']`, honoured by R1 | `test_r1_raw.py`, `test_generator.py` |
| 3 | generator | API clients made half their DNS lookups at 60 s; a backup host resolved once per tick | resolver clock skipped on ticks without a poll; per-tick backup lookup | resolver clock every tick; one lookup per backup run | `test_generator.py` |
| 4 | B13 | budget p < 1e-6 on hundreds of clean control ticks of pack B | a count quantity with a constant same-phase history had a ~0 log scale floored at 2 % | count quantities floor the log scale at their counting noise | `test_b13_budget.py` |
| 5 | B24 | backup hosts: peer p ≈ 1e-17 every night | sparse strata blended with the pm prior forever | own-history floor p ≥ #{ring > s}/(n+1) | `test_b24_calibration_edges.py` |
| 6 | B27 | FP incidents open for days; T4 / T18 / T21 absorbed | habitual lib-4 matches restarted the quiet clock; stale q_inst rows blocked the quiet close | habitual matches do not restart it; q_inst rows older than the quiet window ignored | `test_b27_incident.py` (2) |
| 7 | B27 | 477 of 512 evidence entries were lib-4 info matches | every match appended every tick | ≥ MEDIUM hourly per signature, lower once per (re)opening | `test_b27_incident.py` |
| 8 | B11 | stale `behavior.timing` | active tick without a true gap wrote nothing | NaN descriptors | `test_b11_timing.py` |
| 9 | eval runner (gate 12) | feedback run had MORE control incidents | the simulated analyst used scope 'this'; B23 builds policies only from widened scopes | fp verdicts use scope 'pattern' (eval.md gate 12) | `test_runner.py` |
| 10 | eval metrics (12, 13) | feedback cut / ablation ΔFAR compared unpaired runs | no pairing | paired by (pack, seed) | `test_metrics.py` |

### 10.3 Round-3 fixes

| # | Where | Symptom (evidence) | Root cause | Fix | Test |
|---|---|---|---|---|---|
| 1 | tests | `tests/tests_cadence_part_b.py` never collected | a helper module with a non-`test_` name imported by `test_cadence_invariance.py` | merged into `tests/test_cadence_invariance.py` (`_pb_*` helpers, `run_part_b`) plus an always-collected structural check of its pack | `test_part_b_pack_follows_the_monday_warmup_rule` |
| 2 | lib-4 `signature.rule_match` | health checkers at risk 64–72 after the Runtime's 900 → 60 s switch (78 % from `c2_beacon`, HIGH); pack E: new lib-4 categories on control entities at 60 s, B08 first_seen `cat=beacon` at system tier and jsd p < 0.01 on 99 % of the machine personas' ticks | counter clauses (`http.requests <= 10`, `l4.bytes_up > 3e6`, ...) compare a per-TICK total: a 30-s poller is 30 per 900-s tick and 2 per 60-s tick | canonical grain mode: an additive counter is read as its total per 15-min grain (trailing 900 s at Δt < 900 with the straddling tick pro rata, value × 900/Δt at Δt ≥ 900, newcomers by their observed span); thresholds keep their 900-s meaning; tick mode unchanged (golden) | `tests/test_c2_beacon_cadence.py` (6) |
| 3 | B18 class_rhythm | `scripts/smoke.py` pinned to Mon 09:13 Asia/Shanghai: `erp-prod class:r3` temporal HIGH on the first live tick (S_hi = 21); packs: class_rhythm alarm on 2–2.5 % of class ticks (B, D) and 14 % (E) against 0.005 / day; pack B `class:r3` latched Sat 20:15 → Tue 23:15; pack D new classes alarming from their first days | (a) a slot of a bin with no own history (a class formed during a weekend 900-s phase, or a new class) was scored against the pooled prior, i.e. the other bins' activity (z at the clip); (b) a member counted active in a tick's slot when active anywhere in the tick: a 3600-s warm-up learned the hour's activity as one slot's (1 of 4 vs 4 of 4 in the unit case) | canonical: per-slot membership from `act.slot_events`, every complete slot of a tick scored and learned in time order; a slot scored only when its own bin holds ≥ 3 decayed slots | `tests/engines/test_b18_rhythm_grains.py` (5) |
| 4 | B27 risk opening | pack A seed 0: 58 reopenings of 23 control incidents (median 6 h after the quiet close, 44 of 58 with the risk trigger); pack B: sanctioned health checker 10.40.9.9 reopened ~40 times at risk 90–100, 39 of 42 L15 episodes opened with `risk` | the trigger required the FAMILY hit to be newer than the last incident activity but not the RISK: after a 2-h quiet close the risk decays over days while a family at e_day ≤ 0.1 is an ordinary null event (~0.1 per family and entity-day) | the risk trigger also needs risk ≥ 30 beyond the key's risk at its last incident activity decayed with the slowest B26 half-life (72 h) | `test_b27_incident.py::test_decaying_risk_of_a_closed_incident_does_not_reopen_it_on_a_weak_hit` |

| 5 | B14 changepoint (canonical) | cadence Part B (APPMON_SLOW): 10.20.1.11 and 10.20.1.13 cusum / mcusum acc_alarm from the first live tick to the first live H tick, at 60 s and at 900 s (118 vs 6 alarm ticks for the same two 1-h episodes) | the end-of-warm-up chart restart (§4) runs in `_entity`, i.e. on H decision ticks only; on the live ticks before the first live H tick `_hold_latches` re-emitted the warm-up latch as a live accumulator alarm (every canonical run: up to 3 ticks at 900 s, 59 at 60 s) | `_hold_latches` applies the same restart on the first live tick whatever its type | `test_b14_changepoint.py::test_canonical_first_live_tick_between_h_ticks_restarts_the_charts` (3 cases) |
| 6 | tests (Part B) | `test_part_b_live_60_equals_900` compared B14 acc_alarm TICKS | a latched alarm is re-reported every tick: the same wall-clock episode is 15x the ticks at 60 s | compares B14 alarm episodes (onsets); the null-level test keeps the tick count (0 = 0) | — |

Every new test fails with its fix switched off (checked by monkeypatch). The
tick-mode golden is unchanged (fixes 2 and 3 are canonical-mode only; fix 4
does not trigger in its 48 live ticks).

### 10.4 Results (strict, canonical; packs A and B seeds 0 and 1, C / D / E seed 0)

C, D and E ran with seed 0 only: the whole matrix exceeded ~90 min on the
4 shared cores (one B run took 45 min under contention). "baseline" =
`reports/round2_baseline/`, "HEAD" = `reports/round3_head/` (B0, B1, D0 exact
HEAD runs; A0, A1, E0 the final tree with the §10.3 fixes switched off by
monkeypatch, i.e. HEAD; C0 the 03:54 tree), "final" = `reports/eval_report.json`.
Values are the report's (medians / pooled over runs); per-pack spreads below.

| Gate | baseline | HEAD | final | target |
|---|---|---|---|---|
| 1 overall threat recall / loud / subtle (in deadline) | 0.48 / 0.29 / 0.56 | 0.58 / 0.36 / 0.68 | 0.58 / 0.36 / 0.68 | 0.95 / 1.0 / 0.9 |
| 2 loud TTD (worst listed, ticks) | A/T6b 43 | A/T9 21 | A/T9 31 | 2 |
| 3 FAR ≥ LOW / ≥ MEDIUM per entity-day | 0.206 / 0.138 | 0.209 / 0.129 | 0.237 / 0.152 | 0.2 / 0.05 |
| 3 HIGH+ / CRITICAL (all runs) | 92 / 53 | 63 / 21 | 60 / 19 | ≤ 1 / 0 |
| 3 FAR(E@60 s) / FAR(A@900 s) | 3.38 | 3.10 | 3.10 | [0.5, 2] |
| 4 legit runs within allowed severity | 0.32 | 0.34 | 0.34 | 0.95 |
| 5 notifications per TP incident / class incidents per class legit event | 6.09 / 2 | 6.05 / 2 | 7.05 / 1 (pass) | 3 / 1 |
| 6 incident closed ≤ 2 h after attack end | 0.55 | 0.59 | 0.36 | 1.0 |
| 7 worst median KS D | 0.41 | 0.42 | 0.41 | 0.05 |
| 7 single-tick exceedance e_day ≤ 0.03, cc 900 / cc 60 (x nominal) | 97 / 23,330 | 63 / 3,201 | 64 / 2,818 | [0.5, 2] |
| 7 evidence-CUSUM alarms per entity-day | 1.51 | 1.70 | 1.20 | 0.045 |
| 7 accumulator paths budget / change / temporal_categorical per entity-day | 0.012 / 0.145 / 2.39 | 0.018 / 0.099 / 1.06 | 0.004 (pass) / 0.072 / 1.34 | 0.02 / 0.04 / 0.04 |
| 8 twins confusable / Spearman / T19 chain | 0.43 / 0.11 / 0 | 0.14 / −0.10 / 0 | 0.14 / −0.09 / 0 | 1 / 0.8 / 0.9 |
| 9 role ARI / refit ARI / ID churn / L1–L3 class incidents ≤ LOW | 0.48 / 0.72 / 3 / 0 | 0.58 / 0.66 / 2 / 0 | 0.60 / 0.74 / 0 (pass) / 0.67 | 0.9 / 0.95 / 0 / 1 |
| 10 hit@3 / counterfactual validity | 0.32 / 0.29 | 0.20 / 0.37 | 0.25 / 0.35 | 0.8 / 0.9 |
| 11 per-tick p5–p95 coverage / typical-hours Jaccard | 0.98 / 0.92 | 0.96 / 0.92 | 0.96 / 1.0 | [0.85, 0.95] / 0.8 |
| 12 feedback cut of control incidents ≥ LOW | −0.20 | – | −0.08 | 0.5 |
| 13 ablation | – | – | see §10.6 | |
| 14 max pack-seed wall / live p95 at 35 ent. | 2034 s / 1055 ms | 3183 s / 2164 ms | 2874 s / 1403 ms (contended host) | 360 s / 80 ms |
| 15 exceptions / stale series | 0 / 0 | 0 / 4 (E0 class:r4 common.q) | 0 / 0 (pass) | 0 / 0 |

Per run, HEAD → final (control incidents; "reopened" = extra episodes of
the same id, the lead's open question; ev = evidence-CUSUM alarms per
control entity-day):

| run | FAR ≥ LOW | FAR ≥ MED | distinct ≥ LOW | reopened | openings (distinct + reopened) | HIGH / CRIT | ev |
|---|---|---|---|---|---|---|---|
| A0 | 0.329 → 0.289 | 0.184 → 0.171 | 25 → 22 | 63 → 31 | 88 → 53 | 2/0 → 2/0 | 1.41 → 1.25 |
| A1 | 0.316 → 0.355 | 0.184 → 0.184 | 24 → 27 | 81 → 29 | 105 → 56 | 2/0 → 1/0 | 0.29 → 0.32 |
| B0 | 0.210 → 0.254 | 0.145 → 0.181 | 58 → 70 | 187 → 86 | 245 → 156 | 12/4 → 14/5 | 3.55 → 2.16 |
| B1 | 0.221 → 0.243 | 0.109 → 0.145 | 61 → 67 | 166 → 60 | 227 → 127 | 12/3 → 11/2 | 1.83 → 1.03 |
| C0 | 0.191 → 0.268 | 0.134 → 0.191 | 10 → 14 | 41 → 18 | 51 → 32 | 4/2 → 5/1 | 4.49 → 4.42 |
| D0 | 0.105 → 0.135 | 0.066 → 0.071 | 43 → 55 | 121 → 51 | 164 → 106 | 18/9 → 17/8 | 0.36 → 0.42 |
| E0 | 1.000 → 1.000 | 0.786 → 0.893 | 28 → 28 | 12 → 8 | 40 → 36 | 13/3 → 10/3 | 1.07 → 1.11 |

Reading: the B27 fix halves the reopenings everywhere and cuts the control
openings (what an analyst is notified of) by 10–48 %, but the gate counts
DISTINCT incident ids, and those rise: an FP incident that used to be
revived by old risk and absorb every later alarm of the entity under one id
now stays closed, and a later unrelated alarm opens its own. FAR ≥ LOW is
over target before and after. Threat detection per seed is unchanged in
count; on A0 T1 and T17 flip from detected to missed and on B0 T4b from
missed to detected: all are the same mechanism, an entity at risk
~27–32 before onset (API clients idle at risk 20–30 from identity /
categorical family evidence) that either already has a LOW risk incident
open at onset (the attack then only escalates it to CRITICAL with the
expected axes; eval.md counts an opening, not an escalation) or does not.
T3 on B is now detected by its own alarms after ~1 day (in deadline)
instead of by a risk opening of old risk at +10 ticks; its incident stays
open while the ramp continues (8–12 notifications instead of 1), which is
also why "closed ≤ 2 h after attack end" drops for T3 / T15 / T18.

Smoke: `scripts/smoke.py`'s Runtime pinned to Mon 2026-09-28 09:13
Asia/Shanghai (120 × 3600 s + 192 × 900 s, 16 live ticks at 60 s): HEAD
11 open incidents (5 demo threats, 6 clean keys incl. erp-prod class:r3
temporal HIGH, oa-portal class:r3 LOW, the health checkers 10.40.9.9 MEDIUM
and 10.30.9.9 LOW); final 6 (the 5 demo threats and one LOW categorical
first_seen on 10.30.2.24). Eval smoke pack seed 0: lib-3 60 s (gate 14: 12
s; contended host). mini seeds 0 / 1: 0 exceptions, 0 stale series.

Tests: full suite 2185 passed, 4 skipped. APPMON_SLOW Part B:
`test_part_b_live_60_equals_900` FAILS at HEAD and at final (H rows equal,
B14 episodes now equal, but at 60 s 187 single-tick alarms against 3 at
900 s and 5 vs 1 MEDIUM+ incidents; HEAD 149 vs 4, 7 vs 0). It passed in
the cadence work because the old generator made the 60-s-stepped humans
look alike in both twins and both were equally noisy; with the round-2
generator fix the 900-s twin is quiet and the 60-s t-stream calibration
problem (§10.7 item 2) is exposed. Not marked xfail: it is a real open
issue. `test_part_b_null_levels` stays xfail.

### 10.5 Design constants (lead decision): measured effect

- **β = (0.5, 0.25, 0.25).** e_day of a tick of type τ is q_all n_τ / β_τ,
  so other shares can be replayed from the recorded per-tick e_day and tick
  type (single-tick path; final A0, B0, E0). Control single-tick alarms per
  1000 ticks, design / Q-heavy (0.25, 0.5, 0.25) / n-proportional /
  H-heavy (0.8, 0.1, 0.1): A 4.8 / 5.4 / 5.6 / 3.4, B 23.1 / 26.7 / 26.7 /
  20.6, E 58.7 / 59.0 / 65.5 / 52.3 (nominal ≈ 0.31 at 900 s). Threats'
  first single-tick alarm: unchanged for 21 of 24 (pack, scenario); a Q-heavy
  split loses T1's H-tick alarm at +4 and gains nothing earlier; T9 / T9b /
  T6b move by 4–50 ticks. So the shares move the null rate by ±15–25 % and
  are not what limits loud TTD (the Q grain's own p-values, 1e-3 … 3e-4 for
  the loud T1 replacement on its first three Q ticks, are) or the FAR (the
  ≥ 60x excess is calibration, §10.7).
- **Evidence split 0.5 / 0.5** (ARL 66 d per stream): realised evidence-CUSUM
  alarms 1.20 per control entity-day overall (0.32–4.42 per run) against the
  0.045 budget, i.e. ~25x whatever the split; the stream CUSUM inputs are not
  in the collected series, so the split itself could not be replayed.

### 10.6 Ablation (gate 13) and P2

From the interrupted evaluator's batch (tree of 03:54 = HEAD minus the B13
count floor; every ablation paired with the full run of the same pack-seed
of that tree; `reports/round3_ablation/`): budget (sole detector of B/T4b),
changepoint (B/T4), client_identity (A/T6, T6b, T9b, T16), novelty (A/T13),
rhythm (A/T17), sequence (A/T17) and timing (B/T4b, T10, T14) are each the
sole or main detector of a scenario; attribution, beacon, class_monitor,
likelihood, multivariate and rule_match are not on the ablated pack-seed
(no recall lost; ΔFAR ≥ LOW −0.026, +0.011, +0.002, −0.013, +0.013, −0.013).
B18 is not the sole detector of T21 (T21 is detected with B18 disabled).
Disabling B07 rhythm LOWERS FAR ≥ LOW by 0.079 and makes A/T7 detected: B07
is a net false-alarm source on pack A. Gate 12 on that tree: −0.12 (final:
−0.08) — labels increase control incidents.

P2 (B19–B22, not built): the only direct signal is T20 (lateral access,
expected detector B21 cross_system, axes lateral / xsys): missed on D in
every run (an incident exists on the entity but no built engine emits the
lateral / xsys axis), so B21 would add a gate-1 scenario. Nothing in the
ablations points at B19 (no missed or late scenario is a bimodal-residual
case), B20 (T12 credential stuffing is late, 13–23 ticks against 3, but its
expected detectors B04 / B10 do fire; a session-shape detector might help)
or B22 (role clustering and identification gates fail for reasons listed in
§10.7).

### 10.7 Tuning and design candidates (not changed; evidence)

1. **Recurring HIGH lib-4 matches in a clean warm-up** (L15 backup hosts,
   L7 NAT). `bulk_upload` (HIGH) matches the sanctioned nightly backup
   every night: B26 never habituates HIGH, B28's training trust is 0 on
   those ticks, so B13's committed ring has no trusted night hour, every
   own day / 7-d window is invalid and B13 falls back to the PEERS' level
   (10.20.9.5 bytes_up.7d 6.8 GB against a 20.8 MB "usual", fit_source
   peer, 19 days of history). Risk 80–100 permanently (B0 profile:
   lib4:bulk_upload 41–66 % of it). Options: habituate a HIGH match that
   recurs at the same phase on ≥ N warm-up days, a sanctioned-automation
   allowlist, or keep training trust. L15 is CRITICAL in every run.
2. **60-s t-stream calibration** (pack E, Runtime, Part B). After the
   900 → 60 s switch the (daypart, t, cc=60) meta strata and the cc=60 B24
   strata start empty; p_all includes the accumulators by spec; live rows
   with accumulators ≥ h/2 are kept out of the rings by the evidence cap.
   E0 (interim run, final tree without the B14 / B27 fixes): single-tick
   alarms on 6.7 % of control t-ticks, rising from 13 in the first hour to
   342 in hour 19; the lowest detector is budget_vol / timing / jsd /
   budget_breadth in ~92 % of them (x nominal at p < 1e-3:
   jsd 467, timing 49, budget_vol 48, budget_breadth 35). Options: seed
   cc=60 strata from cc=900 for window statistics (timing, jsd, budget_*
   are defined on 1-h … 7-d windows), or keep accumulators out of the
   t-tick single-tick fusion.
3. **lib-4 per-tick ratio / entropy clauses at 60 s.** After the counter fix
   E0 still shows categories never seen at 900 s: `search` / `api` on humans
   (path_entropy, get_ratio of 1-minute samples), `maintenance` on machines,
   `health`. Option: evaluate lib-4 on Q-window aggregates in canonical
   mode (entropies of merged count maps, ratios of per-grain sums).
4. **Loud TTD at 900 s.** The loud T1 replacement reaches p 1e-38 on the
   first H tick (+4) but only 1e-3 … 3e-4 on the Q ticks before it (the
   conservative H → Q transfer, variance factor 4); §10.5 shows the budget
   shares are not the lever.
5. **Detection metric vs B27 joining** (spec question): an attack on an entity
   with an open LOW FP incident escalates it (CRITICAL, expected axes, an
   'escalate' notification) but eval.md counts only openings (A0 T1 / T17,
   E0 T1' / T12'). Decide whether an escalation with a new expected axis in
   the window is a detection.
6. **Detector calibration at 900 s** (live vs warm-up rings), x nominal at
   p < 1e-3 on A0 control ticks: identity 21, timing 19, spe 17,
   marg_shape_q 13, budget_vol 12, marg_shape 10 (KS: spe 0.41, identity
   0.33). Clean API clients idle at risk 20–30 from identity / categorical
   family evidence (B16 on machine personas; the health checker 10.40.9.9:
   identity p ≈ 2e-4 on many H ticks, 56 % of its risk).
7. **Memory at 60 s** (gate 14): age-based retention of per-tick dict series
   (behavior.score / pm / p 1 d) holds 15x the points at 60 s; E0 derived
   bytes 740 MB at the end, 163 MB / entity extrapolated to 8 d.
8. **Cold start and renumbering** (L6, L8): the renumbered / new IP alarms
   (marg_int single tick, spe) before link seeding or the class prior
   carries it; L8 typing prob never ≥ 0.8 within 3 ticks.
9. **Gate 12**: the simulated analyst's pattern-scope fp labels increase
   control incidents (−0.08 … −0.20); not diagnosed further this round.
10. **Evidence CUSUM**: 25x its budget; the previous evaluator's `evcap`
    experiment (one tick may raise the CUSUM by at most h) was queued but is
    not part of this report.

### 10.8 Reports

`reports/eval_report.{json,html}` (final), `reports/round3_head/` (HEAD),
`reports/round3_ablation/` (gates 12 / 13 on the 03:54 tree),
`reports/round2_baseline/`, `reports/round1/`, `reports/before/`,
`reports/experiment_A_900s_warmup/` (history).

## 11. Round 2 in one place: what changed, results, what remains

"Round 2" here is the whole second development round after the first
evaluation (§8): the cadence-invariant spec v2.1 (§8.1, cadence.md), the W7
tuning fixes (§8.2), the results-neutral performance pass (§9) and the
evaluator rounds 2 and 3 (§10). This section is the summary; the evidence is
in the sections cited. The Chinese design document
(`docs/组织业务系统画像平台设计.md`, library 3 §3.3–§3.4) carries the same results.

### 11.1 What changed

| Area | Change | Where |
|---|---|---|
| Representation | Canonical trailing grains H = 3600 s / Q = 900 s from additive parts (`feature.part`), SetSketch unions (`l4.peer_ids`, `l4.dport_ids`, `tls.ja3_ids`, `act.template_ids`), merged count maps and 15-min slot span features (`act.slot_events`); scored rows only on epoch-aligned decision ticks; midpoint time context; `grain_mode='canonical'` is the default for the pipeline, Runtime, eval and scripts, `'tick'` (golden-tested) for unit tests | cadence.md §2–§5, §17 |
| Baselines and predictives | B03 H anchors plus a native Q current anchor; Q predictive = Q native ⊕ κ_T = 16 pseudo-rows of the H predictive transferred with v = 1 + ((m−1)/m)ω (ω from paired hours, EB-shrunk); B04 / B05 / B06 per grain; four Q detectors (`marg_int_q`, `marg_shape_q`, `t2_q`, `spe_q`; 35 detectors) | cadence.md §6 |
| Sequential detection and budgets | B14 charts on hourly decision rows (h independent of Δt) with latches held between H ticks; identity windows = 4 active H rows with non-overlapping CUSUM steps; single-tick e_day = q_all·n_τ/β_τ over tick types; one evidence CUSUM per stream (S_t, S_h) with a per-stream audit | cadence.md §7–§8 |
| Calibration and decisions | B24 grain strata and pm prior with randomised atoms; B25 winsorised meta tail and release cap (`behavior.trust_evidence`); B07 automation index; corroborated REJECT; B24 own-history floor; B27 quiet-close, lib-4 evidence and risk-reopening rules | §8.2, §10.2, §10.3 |
| Downstream | B29 per-grain replay (≤ 96 H states, per-stream evidence), B30 per-grain bands, API `grains` block, `behavior.degraded` causes, `lib/m_portrait`, B23 profile after refit, narrative fields | §8.2, cadence.md §9 |
| lib-4 | Fresh-only snapshot; additive counter clauses read per 15-min grain in canonical mode (`c2_beacon` no longer matches a 30-s health poller on every 60-s tick) | §4, §10.3 |
| Generator and packs | Continuous-time human sessions, per-flow retransmit fields, resolver clocks; warm-ups end with a phase at the live cadence covering both day types (`packs.warmup_phases`); Runtime plan 120 × 3600 s + 192 × 900 s | §10.2, cadence.md §10 |
| Performance (results bit-identical) | Batched B04, identity background memo, B10 tier reuse, store fast paths and `names_signature`, runner stale-check cache, GC thresholds, retention rules for 8 unbounded lib-3 series: about −20 % wall / CPU on packs A and B, memory under 12 MB per entity at 900 s | §9 |
| Tests | Suite 2118 passed / 3 skipped after cadence M8 → 2185 passed / 4 skipped after round 3; new golden, grain, cadence-invariance, per-fix regression tests; `tests/tests_cadence_part_b.py` merged into `tests/test_cadence_invariance.py` | §8.2, §9, §10.3 |

### 11.2 Results

First evaluation (§8, v2 tick semantics, seed 0) → round-3 final (§10.4,
canonical; A / B seeds 0–1, C / D / E seed 0):

| Measure | §8 "after", A / B seed 0 | round-2 baseline | round-3 final | target |
|---|---|---|---|---|
| threats opening their own incident within deadline | A 1/12, B 1/10 | recall 0.48 | recall 0.58 (raw, any time: 0.75) | 0.95 |
| FAR ≥ LOW / ≥ MEDIUM per control entity-day | A 0.25 / 0.197, B 0.083 / 0.058 | 0.206 / 0.138 | 0.237 / 0.152 | 0.2 / 0.05 |
| HIGH / CRITICAL control incidents | A 10 / 6, B 10 / 5 | 92 / 53 (all runs) | 60 / 19 (all runs) | ≤ 1 / 0 |
| single-tick exceedance at cc 900 (× nominal) | ~500 | 97 | 64 | [0.5, 2] |
| evidence-CUSUM alarms per entity-day | – | 1.51 | 1.20 | 0.045 |
| pack A wall | 850 s | – | 932 s after v2.1 → 740 s after §9 (uncontended); round-3 max pack-seed 2874 s on a contended host | 360 s |

Gates in `reports/eval_report.json`: only 15 (robustness) passes out of the
15; 13 (ablation) is n/a in the final report and was measured on an older tree
(§10.6); 1–12 and 14 fail. Passing sub-checks include window top-1 and
EER_hard, T9 / T9b, L6 linking, T21 class detection, class ID churn 0, the H /
Q portrait band coverage, anchors unpoisoned after attacks, the budget
accumulator path rate and bursty / steady FAR parity.

### 11.3 What remains (ordered by impact)

1. **Recurring lib-4 HIGH matches in clean warm-ups** (L15, L7): design
   decision needed (habituation of recurring same-phase HIGH matches, a
   sanctioned-automation allowlist, or training trust) — §10.7 item 1.
2. **60-s t-stream calibration** after a 900 → 60 s switch; the APPMON_SLOW
   Part B test `test_part_b_live_60_equals_900` fails for this reason and is
   deliberately left failing — §10.7 item 2, cadence.md §17.1.
3. **lib-4 ratio / entropy / concentration clauses per tick at 60 s** — §10.7
   item 3.
4. **Detection metric vs incident joining**: escalation of an already-open LOW
   FP incident is not counted as a detection (spec question for eval.md
   gates 1 and 5) — §10.7 item 5.
5. **Detector calibration at 900 s** (identity 21×, timing 19×, spe 17×,
   marg_shape_q 13×, budget_vol 12× nominal at p < 1e-3; idle API clients at
   risk 20–30) — §10.7 item 6.
6. **Loud TTD ≤ 2 ticks at 900 s** is structurally out of reach under the
   conservative H → Q transfer; β shares are not the lever — §10.5, §10.7
   item 4.
7. **Gate 12** (pattern-scope fp labels increase control incidents), **gate 14**
   (CPU several times over; 60-s memory ~163 MB per entity extrapolated) and
   the identification / class / explanation / poisoning sub-gates — §9.4,
   §10.7 items 7–10.
8. Short warm-ups: B06 commits zi rows scored against hyperprior predictives
   (engines.md B06, cadence.md §17 item 2).

Docs sync note: `scripts/evaluate.py`'s docstring example
`--ablate likelihood,class_monitor` matched no engine (the runner compares
`disable_engines` with the full engine name or the class name, so the job ran
unablated); the example and the `--ablate` help now use full names
(`behavior.likelihood`). The round-3 ablation runs used full names, so their
results are unaffected.

## 12. Evaluation round 4 (evaluator)

### 12.1 Runs and method

- **b20c4da**: the fix agents' tree as committed by the lead (it already
  contains the interrupted first evaluator's fixes: B24's p-score tail floor,
  bocpd's model p, B02's canonical refit stride, the xsys stale-series rule,
  the regenerated golden). Its runs A0, A1, B0, B1, D0, E0 were finished by
  that evaluator's queue on an exact snapshot of the tree; they are the
  "b20c4da" column below (re-scored with the round-4 metrics).
- **round 4 final**: b20c4da + the fixes of §12.2. Every run strict,
  canonical, `scripts/evaluate._job`-equivalent driver (score + pickled
  RunResult for the chain diagnosis) on a snapshot of the final tree, two
  processes at a time on 4 cores.
- The engine chain of every control incident, missed / late threat and legit
  scenario was read from the collected per-tick series and the incidents'
  evidence; three instrumented runs recorded what the series do not carry
  (B08's per-dimension JSD on pack E, lib-4 HIGH match histories on pack A
  seed 1, B28's trust factors on pack A seed 0).

### 12.2 Fixes (each with a regression test that fails without it)

| # | Where | Symptom (evidence) | Root cause | Fix | Test |
|---|---|---|---|---|---|
| 1 | B21 integration (item v) | first_access_system invisible to B25 / B26 / B27 / B29 / B23; xsys default axis 'discovery'; T20's counterfactual held | the kind was in no DISCRETE_KINDS; no token | DISCRETE_KINDS of B25 / B27 / B29, B26 tier weight (org 15, class 8) / axis lateral / decay novelty, B23 family xsys, `FAMILY_DEFAULT_AXES['xsys'] = ['lateral']`, event token `xsys=<system>` and B29 candidate `token:xsys=<system>` (raises the key's cross_system p to its null median) | `test_b21_cross_system.py` (2) |
| 2 | eval gate 8 (item vi) | Spearman(separability, CV recall) undefined on A / B | recall@1 is 1.0 for every individuated persona | graded CV recall: B15 publishes the median held-out margin log L(own) − max log L(other) (`margin`), over individuated personas + twins; recall@1 fallback | `test_metrics.py`, `test_b15_b16_typicality.py` |
| 3 | eval gate 10 (item vi) | L6 / L8 rows counted as hit@3 misses | 'entity' marker | already excluded in b20c4da; now also reported apart | `test_metrics.py` |
| 4 | eval gate 12 (item vi) | 13 of 20 labels tp on A0 | the simulated analyst took the NEWEST incident (a threat is updated every tick) | severity-weighted queue draw (LOW 1 … CRITICAL 8) over unlabelled, unsuppressed incidents active in 24 h, one label per entity per day (eval.md gate 12) | `test_runner.py` |
| 5 | B14 (item iv) | cusum / mcusum latched 28–58 h after 24-h attacks (T9b, T12, T16) | unbounded S drains at −k per hourly row; the clean clock advanced only on rows where NO alarmed chart rose | S ≤ 4h (Crosier ‖S‖ ≤ 4h; the first passage of h, i.e. every null alarm, unchanged) and an end-of-shift test per alarmed chart (reverse CUSUM of "back at the null mean" against "at the shift's estimated mean", ln 1000); a capped chart counts as rising | `test_b14_latch_end.py` (5) |
| 6 | B25 (item iv) | S = 415 / 2146 against h ≈ 6–12 after B/T14, T15 | unbounded evidence CUSUM | S ≤ 2h in B25, the audit's `evidence_path` and B29's replay (MEDIUM, S ≥ 2h, still reachable) | `test_b14_latch_end.py` (2) |
| 7 | B03 / m_baseline (item iii) | Q reference zr variance 2–6, location +1.9 / +2.2 sd (path_entropy, distinct_templates); marg_shape_q pm 191× | `none` (set / map) features were scored at Q against the HOUR's reference predictive; the reference's variance was scaled by the current anchor's ratio | no Q reference for `none` features (p_f = p_cur); t features: Var = σ²_H,ref + (v − 1) σ²_H,cur and the Jensen shift from the same term | `test_b03_grains.py` (2) |
| 8 | B08 (item viii) | T7 missed on A0 / A1 (caught at HEAD only via anti-conservative q_inst) | the smoothed class IDF cannot call a value used by one peer rare in a class < 9 members (A's ERP users are 8, one of them HR) | rare among peers = Jeffreys posterior median of the peers' share ≤ 20 % (`novelty.peer_rare`) → rare_access MEDIUM | `test_b08_novelty.py` (2) |
| 9 | B14 bocpd | 5 of 23 control incidents on A0, ~12 of 57 on B0 | cp ≥ 0.8 was set on N(0, 1) inputs; heavy-tailed hourly inputs read as changepoints (t3: 0.23 alarms/day) | inputs clipped at ±3 (as ψ); alarm level = the Ville bound of bocpd's own budget, pm ≤ 0.005 / 24 ⇔ cp ≥ 0.9915 | `test_b14_latch_end.py` (2) |
| 10 | lib-4 rule_match | auth_bruteforce (HIGH) on humans' login minutes at 60 s | ratio clauses read per tick next to per-grain counters | ratios / averages as counter-weighted means over the grain (`GRAIN_WEIGHTED`) | `test_c2_beacon_cadence.py` (3) |
| 11 | lib-4 rule_match + B08 | E0: 941 jsd accumulator onsets on 4 machine personas; JSD of B08's 'cat' dimension 0.11–0.35 live vs 0.004–0.04 warm-up, all other dimensions ~0 | per-tick entropy / concentration / distinct counts at 60 s describe a minute; a category matched on every 60-s tick of a grain | map statistics from the raw maps merged over the grain (`GRAIN_MAPS`); B08 counts a category dt / 900 | `test_c2_beacon_cadence.py` (2), `test_b08_novelty.py` |
| 12 | B26 | control humans at risk 60–95 from lib-4 alone; risk openings 10 / 21 (A1), 21 / 57 (B0), 8–11 of 13–16 (E0) | HIGH matches of an entity's own irregular recurring activity counted in full until 5 trusted days AND a schedule; one upload counted 3 × 30 (bulk_upload + two composites) | graded weight log10(1/π)/log10 30 with π the entity's trusted per-day match frequency (Beta prior, mean 1/30, 15 pseudo-days) for 'learning' matches and for 'out'-by-schedule matches of an irregular rule; the strongest match per (tick, kill-chain stage) | `test_lib4_high_habit.py` (4) |
| 14 | B09 | pack E seeds 0 / 1: client_impersonation (HIGH) on a control human at 60 s (a stackless 'none' client, risk 0.80) | the surprise S takes its shares over the last 10 TICKS: ten minutes at 60 s | canonical mode: 10 × max(dt, 900 s) of wall clock | `test_b09_client_identity.py` |
| 13 | tests | tick-mode golden | deliberate tick-mode changes (#5, #6, #9, #12) | regenerated (item vii): same 8 incidents, one MEDIUM → LOW; 108 of 291 alarm ticks change (evidence severity after the cap, bocpd) | `test_tick_mode_golden.py` |

Item (i) — B24 against calibrated pm: b20c4da already holds the first
evaluator's p-score tail floor (beyond u B24's p never decays faster than
the detector's pm: a calibrated pm passes through, an uncalibrated score is
still conformalised, and the go-live step is left to the live power
correction). Measured on b20c4da A0, clean control ticks, × nominal at
p < 1e-3: t2 3.4, spe 2.1, timing 4.0, budget_vol 1.2, identity 1.4 (round 3:
identity 21, timing 19, spe 17). No further change.

Item (ii) — detector learners' admission: measured on pack A seed 0 (clean
live control ticks): trust 1 on 71 %, 0 because the key was quarantined on
18 %, 0 by an accumulator ≥ h/2 or the regime on 7 %, reduced by the row's
own evidence on 2.7 % (partial) and 0.8 % (zero). Period admission keeps
the quarantine and would remove only the last 3.5 % — a truncation of the
extreme upper ~1 % of fused q, which changes a Gaussian scale estimate by
< 5 % — while admitting sub-alarm attack rows (and accumulators ≥ h/2) that
the row trust now withholds. The self-selection that mattered is the null
calibration rings', fixed for B24 / B25 in b20c4da; the detector learners
keep row trust (a poisoning defence, not a calibration estimate).
