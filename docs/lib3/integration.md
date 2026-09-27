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
| behavior | B01 feature_vector, B02 peer_group, B03 baseline, B04 likelihood, B05 common_mode, B06 multivariate, B07 rhythm, B08 novelty, B09 client_identity, B10 sequence, B11 timing, B12 beacon, B13 budget, B14 changepoint, B15 identity_model, B16 attribution, B17 entity_link, B18 class_monitor, B23 feedback, B24 calibration, B25 fusion, B26 risk, B27 incident, B28 governor, [B29 explain slot: `_explain_engines()`], B30 portrait |
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

P2 engines B19–B22 are not registered. They stay behind the ablation gate.

Runtime (`Runtime`): `ctx.config` carries tz, the calendar (holidays and
make-up workdays from the generator clock) and `strict`. Warm-up runs with
`training=True` at Δt = 900 s. The live loop runs at the runtime window. Every
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
- `scripts/smoke.py` runs the v2 runtime (warm-up 900 s, then live 60 s) and
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
   (R17.3).
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
