# Behavior Library v2.1 — layered design

## 0. Principles
1. **Engines are coupled only through the store.** Engines exchange data only through MetricStore: series, float32 vector rings, models, events, incidents, checkpoints and health records. Shared maths lives in pure helper modules under `engines/behavior/lib/`:
   - `features.py`: FEATURE_SPEC v2.
   - `detectors.py`: the detector registry (family, instantaneous or accumulator, default axes).
   - `priors.py`: hyperpriors per feature kind.
   - `bayes.py`: NB, BB and Student-t predictives, mid-p, Φ⁻¹.
   - `combine.py`: weighted HMP, randomised conformal p, e_day.
   - `calib.py`: rings, PWM-GPD tail fit, p_from_ring.
   - `seq.py`: time-based CUSUM, Bernoulli and exponential-evidence thresholds; stationary tails; AR(1) prewhitening.
   - `gating.py`: commit, hold, checkpoint, rollback, release and rebase.
   - `timebins.py`: tz, calendar, bin48/bin168, 15-minute slot, daypart, cadence class.
   - Also `featcache.py`, `sketch.py`, `ppm.py`, `evt.py`, `robustcov.py` (OAS, C-step, RBC, conditional imputation), `template.py` (Drain-lite), `stack.py`, `names.py`, `stages.py`, `classkeys.py`, and `replay.py` (deterministic recompute of stateful detectors).

   No engine imports another engine module.
2. **Time is wall-clock time, not ticks.**
   - Features are per-minute rates with explicit exposure. Half-lives, delays and refit periods are in seconds (`Engine.period_s`).
   - Every sequential statistic gets its threshold from a wall-clock false-alarm target: `arl_ticks = ARL_days·86400/Δt`.
   - Presence runs on a fixed 15-minute slot clock.
   - Conformal rings are Mondrian-stratified by daypart × cadence class.
   - Severity is `e_day = p·86400/Δt`, the expected number of equally extreme null ticks per entity-day.
3. **Absence is data, from end to end.**
   - Raw engines stamp `ts = ctx.now` and emit only for entities that had observations.
   - `add_raw` maintains first_seen and last_seen.
   - Instant derived metrics are emitted only when every input is fresh (D1). Window metrics use a zero-filled grid (D0).
   - feature_vector turns stale counts into 0 and stale ratio, average or bounded values into NaN.
   - `feature.active = 1` if and only if the entity had an observation in (now−Δt, now].
4. **Learn late, from trusted ticks, and reversibly.**
   - A learner commits row t−D, where D = max(4 ticks, 600 s), with weight trust(t−D), unless the entity is quarantined.
   - Held rows wait for the governor.
   - Learners write geometric checkpoints. The governor can `release`, `rebase_from` or `rollback_to` an estimated onset τ̂.
5. **Scores become calibrated p-values, and decisions spend a budget.**
   - Detectors write raw scores together with the axes that drove them.
   - calibration writes randomised, stratified p-values for each entity.
   - fusion combines them with weighted HMP and re-calibrates the combination per entity.
   - The evidence CUSUM runs only over instantaneous detectors. Each accumulator has its own time-based threshold.
   - Only the incident engine notifies.
6. **Hierarchy, with classes as first-class entities.**
   - Estimates back off along entity → role class (per system) → system → org → hyperprior.
   - Classes have two levels: role, then individual sub-class.
   - Every class is a pseudo-entity `class:<id>`, whether it is a dynamic role, a static CIDR class or a synthesised pool. Each has its own aggregate baseline, detectors, incidents, risk and portrait.
7. **Identity is absolute.** Identification, attribution and linking score a window under each candidate's own models. They never use behavior.z, which is normalised to each entity itself.
8. **Everything is observable.**
   - Engine errors, stale inputs, degraded detectors and calibration drift are recorded and exposed.
   - A detector with a stale input writes NaN, never p = 1. Fusion excludes NaN and marks the family as degraded.
   - Tests and eval run with `ctx.config['strict'] = True`, which re-raises errors.

## 1. Per-tick data flow
Layer order is raw, derived, behavior, signature. Within a layer, engines run in the listed order.

**raw**
- l2l3, l4flow*, http*, tls*, dns*, probe.
  - `*`: full sets of up to 64 entries plus `__other__`, weighted aggregated records, `ts=now`.
- action_token (R2): act.tokens, act.stream (with stack_id), act.stream_frac, act.events (zero-filled), act.objs, act.rare_events.
- client_stack (R3): client.stack_set, client.stack_events (with timestamps).

**derived**
- aggregation, periodicity, trend (D0): zero-filled wall-clock grid, kind `window`.
- ratio, entropy, graph (D1): emitted only when fresh; ratios carry their `.n`; the seen-set decays over 30 days.
- session (D2): duty from a 24-hour grid; think_time only when fresh.

**behavior**

| Engine | Writes |
|---|---|
| B01 feature_vector | feature.vec, nat, expo, active, tctx, sketch (+ virtual `feature.<name>`) |
| B02 peer_group @16 or 6 h | model.class (role / sub / static / pool), class profiles, cold-start typing |
| B03 baseline | model.baseline (current and reference anchors, class, system, org); checkpoints |
| B04 likelihood | behavior.z / zr / pf; scores marg_int, marg_shape, peer |
| B05 common_mode | behavior.zi, behavior.common.* (LOO, eligible axes only) |
| B06 multivariate | t2, spe, behavior.wh, model.density, model.groups |
| B07 rhythm | offhours, silence (15-minute slot clock) |
| B08 novelty | novelty, novelty_rate, jsd; events first_seen, rare_access, class_adopted |
| B09 client_identity | client; events client_impersonation, client_change |
| B10 sequence | seq, dwell (PPM with entity → role → system backoff) |
| B11 timing | timing, behavior.timing.* |
| B12 beacon | beacon (Gamma renewal LRT) |
| B13 budget | budget_vol, budget_exfil, budget_breadth |
| B14 changepoint | cusum, mcusum, bocpd, creep (on zr against the reference anchor); cp.onset |
| B15 identity_model @96 or 24 h | model.identity, separability (absolute windows, CV) |
| B16 attribution | identity; events identity_mismatch, unknown_identity |
| B17 entity_link | model.link; events entity_resolution, possible_impersonation, shared_ip |
| B18 class_monitor | class_int, class_shape, class_rhythm, class_novel, class_coherence (at `class:<id>`) |
| P2 B19–B22 | mixture, session, cross_system, action_embedding |
| B23 feedback | model.feedback |
| B24 calibration | behavior.p (randomised, Mondrian, GPD tail) |
| B25 fusion | p_family, q_inst, q_all, e_day, evidence, alarm |
| B26 risk | behavior.risk (entity, class, system) |
| B27 incident | Incident objects and incident events (entity and class) |
| B28 governor | trust, trust_prov, quarantine, regime, model.control (rollback, release, rebase) |
| B29 explain | incident.explanation (faithful replay counterfactual) |
| B30 portrait @8 or 2 h | portraits for each IP and each class; profile versions |

**signature**: unchanged. risk, incident and governor read its matches with a lag of one tick, through the indexed `matches(since)`.

## 2. Sublayers
- **Representation (B01).** Nothing in B01 learns.
- **Shared modelling.**
  - B02 publishes classes.
  - B03 publishes the entity, class, system and org predictives with two anchors.
  - B15 publishes the identity metric and the calibrations for the LLR of each modality.
  - Every detector engine B04–B22 owns one small model. It scores first against its model as of the last commit, then commits gated rows.
- **Identification (B04–B22).** B04–B18 are P0/P1; B19–B22 are P2 and enabled only after the ablation gate.
- **Decision (B23–B30).**

## 3. Trust, gating, rollback, bootstrap
```
e_inst(t) = q_inst(t)·86400/Δt                      # meta-calibrated instantaneous evidence, per day
trust_prov(t) = clip(log10(e_inst/0.1), 0, 1)        # 0 if rarer than once per 10 days, 1 if at least daily
               × [no alarm at t] × [no discrete finding ≥ MEDIUM at t]
trust(t) = trust_prov(t) × [no open incident] × [regime ∉ {SUSPECT, DRIFTING}] × [every accumulator < h/2]
quarantine(t) = incident open OR regime ∈ {SUSPECT, DRIFTING}
ctx.training ⇒ trust = trust_prov = 1, unless a lib-4 match ≥ HIGH exists for (entity, t)
```
**Learner rule (lib/gating.py).**
- At tick t, commit row t−D with weight trust(t−D) if quarantine(t−1) = 0; otherwise hold it.
- Checkpoint the state at most hourly. Keep checkpoints at ages {1, 2, 4, 8, 16, 32, 64, 128, 168} hours, as float16.

**Governor actions (model.control).**
- **SUSPECT with onset τ̂ older than the commit frontier** → `rollback_to = τ̂ − Δt`. Each learner restores its latest checkpoint at or before rollback_to, replays the committed rows in (checkpoint, rollback_to] with their recorded trust, and holds everything after.
- **RETURNED** → `release = [τ̂, t]`: held rows are committed with trust_prov.
- **ACCEPTED** → `rebase_from = τ̂`: version + 1, and the new regime's rows are committed with trust_prov.
- **REJECTED** → held rows are discarded and the entity is frozen.
- Rollbacks are limited to one per entity per hour and reach back at most 7 days.

**Anchors.**
- The current anchor's mean may move by at most 0.1σ15 per day, where σ15 is the predictive sd for a 15-minute exposure. This keeps the cap independent of cadence.
- The reference anchor commits with a 24-hour delay, only rows with trust = 1 and with no incident or regime within ±24 h, and is capped at 0.03σ15 per day. It is replaced by the golden anchor (median of up to 4 weekly snapshots from low-risk weeks) once one exists.
- The likelihood uses p = min(1, 2·min(p_cur, p_ref)). Changepoint uses zr, measured against the reference anchor. A creeping current anchor therefore cannot cancel the signal.

**Cold start.** There is no deadlock.
- Hyperpriors (priors.py) make the first predictives wide, so early p-values are large rather than garbage.
- During training, trust is 1.
- In live operation, a new entity is scored against its role class and system through backoff. A benign newcomer (L8) therefore gets trust 1 and matures, while an attacker (T11) gets trust 0 and stays quarantined.
- A genuinely novel but benign entity reaches ACCEPT through the `new_entity` regime type after 3 days with no malicious-type evidence, or through a label.

## 4. Scoring → decision
**Detector families and kinds** (lib/detectors.py; the axes default is shown, and engines refine axes per tick from their contributions):

| Family | Instantaneous detectors | Accumulator detectors | Default axes |
|---|---|---|---|
| intensity | marg_int, t2, class_int | budget_vol | volume |
| shape | marg_shape, spe, class_shape, mixture* | – | shape |
| peer | peer, class_coherence | – | by group |
| temporal | – | offhours, silence, timing, class_rhythm | temporal |
| categorical | novelty | jsd, class_novel | categorical (exfil / privilege when the value is external, upload-dominant or sensitive) |
| breadth | – | budget_breadth, novelty_rate | breadth |
| exfil | – | budget_exfil | exfil |
| sequence | seq, dwell, session* | – | sequence / credential |
| identity | identity, client | – | identity |
| change | – | cusum, mcusum, bocpd, creep | by feature group |
| c2 | – | beacon | c2 |
| xsys | cross_system* | – | discovery |

`*` = P2.

**Fusion.**
1. `p_family = wHMP(p_d)`, excluding NaN (degraded or unscored) inputs.
2. `p_inst = wHMP(instantaneous families)` and `p_all = wHMP(all families)`, with family weights from feedback × calibration health.
3. Meta-calibration: each entity has a ring stratified by daypart × cadence, plus a GPD tail, giving `q_inst` and `q_all`.

**Decision paths.**
- **Single-tick:** `e_day(q_all) ≤ 0.03`.
- **Evidence CUSUM:** `S = max(0, S − ln q_inst − 3)`, alarm at `S ≥ h(Δt)`, with `h = (ln(ARL) − 3.07)/0.94` and ARL = 33 days in ticks.
- **Accumulators:** each accumulator's own `acc_alarm`, from its time-based threshold.

**Severity.**
- `e_day ≤ 0.03` → LOW; `≤ 3e-3` → MEDIUM; `≤ 3e-4` → HIGH candidate; `≤ 3e-6` → CRITICAL candidate.
- HIGH additionally needs corroboration: at least 2 axes each at family e_day ≤ 0.03 within 4 ticks, or 2 consecutive ticks at ≤ 3e-3, or a discrete finding ≥ HIGH.
- CRITICAL needs 3 or more axes, or a lib-4 match ≥ high.

**Reading rules** (based on axes, not on detector names).
- **Volume only.** If every significant contribution has axes ⊆ {volume}, severity is capped at MEDIUM for an entity and LOW for a class.
- **Self / peer 2×2.** Self abnormal and peer normal: down one level. Both abnormal: up one level.
- **Common-mode flag.** It downgrades volume, transport and app-error evidence by one level. It never affects categorical, identity, c2, exfil or temporal evidence.
- **Schedule shift.** Temporal evidence explained by a detected schedule shift is capped at LOW.

**Null alarm budget per entity-day.** These are design targets and are enforced by eval gate 7.

| Path | Target |
|---|---|
| single-tick | 0.03 |
| evidence CUSUM | 0.03 |
| change family | 0.02 |
| rhythm (off-hours, silence) | 0.015 |
| budget | 0.01 |
| identity CUSUMs | 0.01 |
| beacon, timing, jsd, novelty_rate | 0.02 |
| **Total ≥ LOW** | **≈ 0.135** |

- After merging into incidents, the gate allows ≤ 0.2 incidents ≥ LOW per entity-day.
- MEDIUM runs about 10× lower, ≈ 0.014, against a gate of 0.05.
- HIGH with corroboration is ≲ 1e-4 per entity-day, so the expected count is ≈ 0.12 over 1225 clean entity-days.

## 5. Class as entity (B18)
For every class, B18 builds an aggregate row each tick. Classes are role classes with 2 or more members in a system, static CIDR classes with 3 or more members, and pool classes. The row holds:
- member-summed per-minute counts;
- pooled ratios Σk/Σn;
- the fraction of members active;
- the union of member tokens.

The class aggregate has its own conjugate baseline (model.classagg) with the same two-anchor gating. Its detectors are class_int, class_shape, class_rhythm, class_novel (adoptions of new values across members, weighted by external, upload-dominant, sensitive or new-eTLD+1 status) and class_coherence (a binomial test on members that deviate in the same direction).

Class reading rules:
- A coherent intensity-only shift is capped at LOW (for example L1, the month-end surge).
- The same holds for a system-wide app-error or transport shift (L10).
- A risky adoption (T21, a compromised pool) or a class shape change is graded by its p-value.

Per-entity novelty discounts for adoption cannot hide class-level risk, because class_novel scores the adoption itself.

## 6. Cadence handling
| Quantity | Treatment |
|---|---|
| Features | Per-minute rates with explicit exposure; bounded features are gated by exposure. |
| Commit delay, half-lives, refit strides | Seconds (`Engine.period_s`). A refit runs when either its interval in ticks or its period in seconds is reached. |
| CUSUM, MCUSUM, identity CUSUMs, evidence CUSUM | `h = seq.h_for(kind, k, ARL_days·86400/Δt)`. Per-feature CUSUMs are AR(1)-prewhitened. |
| Rhythm | 15-minute slots. Sub-slot activity is resolved from act.stream timestamps, and the current slot is updated provisionally. |
| Conformal rings | Strata = daypart(4) × cadence class {60, 300, 900, 3600}. A new stratum blends with the model p (pm) and the class ring until it holds 64 entries. |
| Severity, trust, risk | Based on e_day, so they do not depend on cadence. |

## 7. Cost and memory envelope
Costs are per tick for 40 entities, from measured primitives.

| Engine | ms | Engine | ms |
|---|---|---|---|
| B01 | 12 | B11–B13 | 6 |
| B02 | 1 | B14 | 3 |
| B03 | 7 | B15 | 5 |
| B04 | 6 | B16 | 10 |
| B05 | 1 | B17 | 1 |
| B06 | 5 | B18 | 3 |
| B07 | 1 | P2 | 6 |
| B08 | 3 | B23–B28 | 9 |
| B09 | 1 | B29 | per incident |
| B10 | 5 | B30 | 5 |

- The total is ≈ 94 ms per tick at 40 entities and ≈ 48 ms at 20.
- The smoke warm-up (180 ticks at 900 s, 20 entities) spends ≈ 9 s in library 3, against a target of ≤ 12 s.
- Store queries are indexed at O(log n + k), ≈ 2 ms per tick in total. The current linear scan costs ≈ 1.6 ms per call, about 600 ms per tick at 400 calls.
- Memory per entity at steady state (60-s ticks):
  - long vector rings (8 days): 3.2 MB;
  - short rings: 1.6 MB;
  - raw and derived objects under retention (slots dataclasses): ≈ 1.5 MB;
  - act.stream and sets: 1 MB;
  - models: 2.5 MB;
  - checkpoints: 1.2 MB.
- That totals ≈ 11 MB per entity, against a gate of ≤ 12 MB.

## 8. Requirement traceability
| Requirement term | How it is made concrete | Engines | Eval gate |
|---|---|---|---|
| Multi-dimensional combination of raw and derived metrics | 52 features in 9 groups, an 80-dim sketch over 5 namespaces, correlation-aware T²/SPE, HMP fusion | R1–R3, D0–D2, B01, B04, B06, B25 | 1, 7, 13 |
| AI learning | Conjugate hierarchical Bayes with empirical Bayes, HDBSCAN hierarchy, OAS/WCCN-LDA metric learning, PPM, BOCPD, renewal LRT, stacking and isotonic feedback, optional PPMI-SVD/NMF embeddings | B02, B03, B10, B12, B14, B15, B22, B23 | 8, 9, 12 |
| Concrete profile of ONE user | Per-IP models plus a versioned portrait in natural units | B30 plus all model.* | 11 |
| Generalised profile of ONE class | Role and sub-class hierarchy, class aggregate models and detectors, class portraits | B02, B18, B30 | 9, T21, L1 |
| Fine-grained | Templated action tokens, client stacks, sub-tick timing, 15-minute × 168 rhythm, object-level HLL, per-token grammar | R2, R3, B07, B10, B11, B13 | T7, T8, T10, 11 |
| Distinguish one user or class from others | Cross-validated EER/recall, confusable lists, open-set attribution, linking | B15, B16, B17 | 8 |
| Precise detection of deviation from the daily profile | Exact exposure-aware predictives, dual anchors, calibrated false-alarm budgets, accumulators for slow attacks | B04–B14, B24, B25 | 1–3, 6, 7 |
| Advanced | Conformal + EVT, HMP, CUSUM/MCUSUM/BOCPD, PPM, HDBSCAN, OAS-WCCN-LDA, Good–Turing, Fellegi–Sunter, renewal LRT, reversible governance | – | 7, 13 |
| Complete | Every behavioural axis has at least one engine and at least one scenario (volume, shape, correlation, time-of-week, sequence, timing, resources, client, identity, class, cross-system, cumulative, change). The ablation gate proves each engine necessary. | – | 13 |
| Comprehensive | Entity and class levels; loud and subtle threats; legitimate changes | – | 1–4 |
| Intelligent | Legitimate-vs-attack reasoning, rollback, feedback learning, active label queue, adaptive half-life, ACI, calendar self-healing, faithful counterfactual explanations | B23, B28, B29 | 6, 10, 12 |

## 9. Why statistical-first rather than deep models
**The data are small.** There are about 20–40 entities and roughly 10³–10⁴ sessions per entity per week.

**The runtime constraints are tight.**
- CPU only, with a budget of < 100 ms per tick.
- Calibrated per-entity tail p-values down to 1e-5, which needs likelihoods or conformal scores with EVT.
- Rollback to arbitrary onsets, which needs cheap sufficient statistics.
- Faithful explanations.

**The evidence favours simple models.**
- PPM is a near-optimal predictor for small alphabets (Begleiter et al., JAIR 2004).
- Studies of log anomaly detection report that n-gram and count baselines match deep models once data leakage is controlled (Le & Zhang, ICSE 2022; Landauer et al., 2023).
- PPMI-SVD is equivalent to SGNS word2vec (Levy & Goldberg, 2014).

Learned embeddings are therefore offered as the P2 modality B22, and are enabled only if the ablation gate shows a gain. The upgrade path for deployments with 10³ or more entities replaces B22 or B15 internals with a pre-trained template encoder or a contrastive entity encoder behind the same store contract.

## 10. Core and pipeline fixes
- `Pipeline.run_tick(obs, now, training, dt)` sets `ctx.window_s = dt` (currently fixed at orchestrator.py:33).
- The live loop uses `now = gen.vt` (build.py:115).
- `Engine.safe_run` (engine.py:76-84) is extended:
  - it records error_count, last_error_ts and the head of the traceback, and calls `store.put_health`;
  - it re-raises in strict mode;
  - it also runs when `period_s` has elapsed.
- The orchestrator writes `ops.engine_health`.
- The dataclasses use `slots=True`.
- The store gains retention, indexes, vector rings, checkpoints and health (see shared_contract D).
- Every learner persists its state via `put_model` at refit or checkpoint time, so the pipeline can restart.