VERDICT (re-verified against the code this session). The current library 3 does not meet the requirement. The root causes still hold at the cited lines:
- Stale carry-forward: store.py:106-120 returns the latest value whatever its age, and feature_vector.py:52 defaults a missing source to 0.
- The derived layer re-stamps stale inputs with ts=ctx.now: ratio.py:36-58, entropy.py:27-50, graph.py:31-67, aggregation.py:40-41, periodicity.py:43, trend.py:38 and session.py:40-60. As a result duty_cycle is only ever 0 or 1 and think_time is always 0.
- A relative floor on log values: anomaly.py:102-104.
- Nothing gates what is learned: baseline.py:51-61 and fingerprint.py:42-46.
- The seasonal booster cannot fire on its own and works in UTC: anomaly.py:74-77, 146-157.
- The IsolationForest is refit on every run: anomaly.py:117-139.
- Clustering uses KMeans with thresholds on z-scores: clustering.py:63-111.
- The sequence model is keyed on a churning name with a fixed alphabet of 8: sequence.py:60, 88-90.
- Separability is a nearest-neighbour cosine between medians: fingerprint.py:80-94.
- Categorical sets are truncated to the top 8 or 12: http.py:102, tls.py:78, dns.py:66.
- Clock and cadence are skewed: orchestrator.py:33, build.py:104-115.
- Engine errors are swallowed silently: engine.py:76-84.
- Every events()/matches() call scans the whole deque: store.py:131-145.
- Storage is 20000-point object deques with no limit by age: store.py:33-35.

REVISION. This spec fixes all 11 'missing', 24 'incorrect/risky' and 9 'contract conflict' findings from the critic.
(1) Freshness chain from end to end.
- New derived engines D0 (window metrics computed on a zero-filled wall-clock grid) and D1 (instant metrics emitted only when every input is fresh). The session engine D2 is fixed.
- An entity counts as active when it has an observation in the ingestion tick. The old Δt/2 rule is removed.
(2) Cold start.
- Weak organisation-level hyperpriors per feature kind (lib/priors.py).
- trust is set to 1 only during ctx.training, and even then not for entities with a library-4 hit at HIGH or above.
- Forcing trust=1 for immature entities during live operation was rejected, because it would absorb T11. Instead, a new entity is scored against its class and system through backoff, and a separate acceptance path handles new entities.
(3) Learning is reversible. Learners keep geometric checkpoints, a commit journal and a provisional trust. model.control supports rollback_to, release and rebase, so detections slower than the commit delay D no longer leave attack rows in any model.
(4) Classes are entities. A new engine, B18 class_monitor, gives dynamic role classes, static CIDR classes and pool classes their own aggregate baseline, rhythm, vocabulary, detectors, incidents, risk and portrait.
- common_mode is now leave-one-out and needs at least 3 other members.
- It is limited to the volume, transport and app-error axes, and never applies to categorical, identity, c2 or exfil evidence.
(5) Two-level hierarchy. Classes are role, then individual sub-class, each found by HDBSCAN on its own distance. The role is the class used for backoff, class_monitor and ARI.
(6) Identity uses absolute representations and scores each window under each candidate's own models. behavior.z is never used for identity.
(7) Calibration and fusion.
- Conformal p-values are randomised and stratified by daypart × cadence class.
- Families are combined with a weighted harmonic-mean p (HMP) instead of ACAT, then re-calibrated per entity.
- Measured this session: ACAT([1e-3, 0.999, 0.5]) = 0.5, while HMP gives 0.003. Under equicorrelated dependence (ρ = 0.64) HMP's false-alarm rate is 1.095× nominal at α = 1e-3 and 1.005× at 1e-4.
(8) False-alarm budgets are set in wall-clock time.
- e_day = p·86400/Δt.
- The evidence CUSUM runs only over instantaneous detectors, with k = 3. Its average run length (ARL) was solved exactly: ARL ≈ 21.6·e^{0.94h}, with h = 5 giving 2393 ticks and h = 8 giving 40263 ticks. The target is 33 entity-days, so h = 5.31 at 900 s and 8.19 at 60 s.
- Accumulating detectors have their own time-based thresholds.
- HIGH requires corroboration from a second source.
- The raw budget of alarms at LOW or above totals ≈ 0.14 per entity-day.
(9) Risk.
- Evidence is the 'excess surprise over daily expectation', so it does not depend on cadence.
- Evidence within one episode saturates.
- L_ref is a fixed 60 rather than being poisonable. The simulated null gives mean L ≈ 7 and p99.99 ≈ 17–19 at both 60 s and 900 s, which keeps risk ≤ 27.
(10) Governor.
- Closing an incident no longer depends on risk.
- Time spent stationary in a new regime counts as evidence. A single-entity intensity change is accepted after 1 day (logit 3.46), a shape change after 7 days (logit 2.46), and a categorical change needs a label (logit 1.96).
- A ramp-rate rule applies: 0.05/day in log terms counts as legitimate.
- No entity can be locked out permanently.
(11) Poisoning. The current anchor is capped at 0.1σ/day. A reference anchor (0.03σ/day, 24-hour delay) feeds p = 2·min(p_cur, p_ref) and the changepoint residuals zr.
(12) Numerical fixes.
- The Bernoulli CUSUM clips p1 and runs on a 15-minute slot clock.
- The Hellinger sketch formula is corrected.
- Hotelling uses the prediction scaling, and missing dimensions are imputed conditionally.
- The beacon test is now a Gamma renewal LRT with a finite-n Monte-Carlo null. Measured: with ±30% jitter the median p is 1.7e-8 at n = 12. The χ² asymptotic is 2–5× anti-conservative on a Poisson null. The Rayleigh median p is only 0.14 at n = 40.
- The likelihood tests now use scipy-verified tails: Poisson(22) P(X≤3) = 5.7e-7 and P(X≤1) = 6.4e-9; BB(200, mean 0.1, c = 50) sf(99) = 1.3e-8, and 7.7e-5 at c = 20.
(13) Store.
- Indexed events and matches per entity with since and kinds filters, O(log n).
- float32 vector rings with virtual scalar views, so the 45 compat series no longer copy 20000 objects each.
- A retention table, first_seen/last_seen, checkpoints and a health record per engine.
- Strict mode for tests and eval.
(14) Why 'SOTA' here means this design. The architecture document maps the requirement terms to engines and eval gates. The dropped list evaluates DeepLog, LogBERT, AE/VAE, contrastive and GNN alternatives. An optional P2 learned modality, B22, adds PPMI-SVD and Poisson-NMF embeddings.
(15) Generator. Five packs, one scenario per entity per pack, each timezone in its own run, one cadence pack, aggregated observations, and projected CPU of ≤ 5 minutes per pack-seed.

COST. Library 3 costs ≈ 2.3 ms per entity per tick: ≈ 48 ms at 20 entities and ≈ 94 ms at 40. The smoke warm-up spends ≈ 9–10 s in library 3.
- identity_model: LDA on 600×64×20 measured 29 ms, so a 96-tick or 24-hour stride costs ≈ 5 ms per tick amortised.
- OAS on 600×64 measured 1.9 ms.
- Memory target: ≤ 12 MB per entity in steady state.

Scratch evidence: /tmp/claude-0/-home-user-appmonitor-core/e496f878-8276-5793-a91f-1f6d0128e252/scratchpad/rev_checks.py (tails, ACAT/HMP size), rev_checks2.py (exact exponential-CUSUM ARL), rev_checks3.py (Siegmund thresholds, null risk simulation), rev_checks4.py (Gamma LRT, Rayleigh, LDA/OAS timing, Bernoulli increments). The earlier bench_redesign.py and bench2.py are in the same folder.