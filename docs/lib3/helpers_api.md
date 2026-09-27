# Behaviour library helpers — public API (frozen contract)

Package: `backend/app/engines/behavior/lib/` (import as `from app.engines.behavior.lib import bayes`).
Every engine is written against the signatures below. The module docstrings hold the exact maths, and this page is the index. **Status**: **impl** means the module is implemented and tested (`tests/lib/test_lib_data.py` plus `tests/lib/test_<module>.py`). **stub** means the signature, dataclasses and maths are frozen, but the body raises `NotImplementedError` until the implementing wave lands. After wave 0 only `replay` and `featcache` are stubs.

## 0. Conventions (apply to every module)

- **Purity.** Helpers are pure functions or small dataclasses. Only `gating`, `replay` and `featcache` take a `MetricStore`, and they receive it as an argument. No helper imports an engine.
- **Store API names** (contract D) used by helpers and assumed by engines: `add_vec`, `vec_tail`, `vec_since(s, e, name, since) -> (ts[k], M[k, d])` (ts ≥ since), `vec_at(s, e, name, ts) -> row written at exactly ts | None`, `latest_fresh`, `put_model`, `get_model`, `model_version`, `put_checkpoint(s, e, learner, ts, blob)`, `get_checkpoint(s, e, learner, at_or_before) -> (ts, blob) | None`.
- **Governance scalars.** B28 writes `behavior.trust`, `behavior.trust_prov` and `behavior.quarantine` as **1-element float32 vec rings** (`add_vec`). `gating` reads them that way.
- **NaN policy.** NaN means *undefined, unscored or degraded*. It is never silently turned into p = 1 or 0. Stale counts are a true 0. A stale ratio, average or bounded value is NaN. Combiners drop NaN inputs, and an all-NaN input gives NaN.
- **Numerics.** Tails are computed on their own side (sf, not 1 − cdf). p is clipped to `[1e-300, 1]` before logs or inversion, and `phi_inv` clips u to `[1e-15, 1 − 1e-15]` (|z| ≤ 7.94). Everything is float64, except rings stored as float32.
- **Time.** Seconds are wall-clock UTC epochs. Thresholds come from ARLs in days through `seq.arl_ticks(arl_days, dt_s)`. Severity uses `combine.e_day(p, dt_s)`.
- **Determinism.** Randomisation uses `combine.seeded_uniform(*keys)` (blake2b), never `random` or `hash()`, so replay is bit-identical.

## features — FEATURE_SPEC v2 (impl)

Constants:

| Name | Value |
|---|---|
| `FEATURE_SPEC_V2` | 52 × `(name, source, kind, n_source, group)` |
| `FEATURE_DIM` | 52 |
| `FEATURE_NAMES_V2` | list of the 52 names |
| `FEATURE_INDEX` | `{name: idx}` |
| `FEATURE_KIND` | `{name: kind}` |
| `FEATURE_GROUP` | `{name: group}` |
| `FEATURE_NSRC` | `{name: n_source}` |
| `FEATURE_SOURCE` | `{name: source}` |
| `GROUP_ORDER` | the 9 groups: volume, breadth, app, dns, tls, timing, transport, probe, comp |
| `GROUPS` | `{group: [idx]}`, a partition of 0..51 |
| `KEY_FEATURES` (12) | the CUSUM-bank features |
| `KEY_FEATURE_IDX` | indices of the key features |
| `CLR_FEATURES` / `CLR_IDX` | the 8 comp columns |
| `EXPOSURE_CHANNELS` | `[http, dns, tls, flows, probe]` |
| `EXPOSURE_SOURCE` | `{channel: metric}` |
| `VEC_TX` | per-feature vec-transform overrides: `updown_log` → identity, `req_per_session` → log1p |
| `KINDS` | the allowed kinds |

**Source encoding.**
- `"metric"` reads a fresh metric.
- `(num, den)` means num/den. For a ratio kind it means k = num and n = den.
- `("logratio1p", a, b)` means log((a+1)/(b+1)).

For a ratio with a plain metric source, the metric is a fraction and k = fraction·n. `n_source` is `None`, a metric, or a tuple of metrics that are summed.

- `transform(kind, v, n=None, dt_s=60, *, tx=None) -> (vec, nat)` applies the per-kind transform.
  - count/bytes: vec = log1p(v·60/dt), nat = raw v. Stale gives (0, 0).
  - ratio: vec = logit((k+.5)/(n+1)), nat = k/n. Stale or n ≤ 0 gives NaN.
  - avg: vec = log(max(v, 1e-3)). Stale, v < 0 or n < 1 gives NaN.
  - bounded: vec = logit(clip(v, 1e-3, 1−1e-3)). Requires n ≥ 5, otherwise NaN.
  - gauge/window: identity, or `tx`. Stale gives NaN.
  - ctx: (NaN, v).
  - clr raises an error; use `clr`.
- `transform_feature(idx, v, n, dt_s)`: `transform` with the kind and `VEC_TX` looked up for feature idx.
- `clr(counts[8], dt_s=60) -> (vec[8], nat[8])`: CLR of per-minute rates, log(x·60/dt + 0.5) − mean. A total of 0 gives all NaN; nat = shares.
- `feature_inputs(idx, get) -> (v, n)`: resolves a source formula and exposure from `get(metric) -> fresh value | None`. A declared `n_source` that is stale gives n = 0, so the feature is NaN.
- `compute_features(get, dt_s) -> (vec[52], nat[52])`: the whole B01 numeric row from a freshness-aware getter.
- `exposure(get) -> {channel: n}`: `feature.expo`. Stale gives 0.
- `source_kind(src)` returns `'metric' | 'div' | 'logratio'`. `source_metrics(src)` returns the metric names a source reads, and `all_source_metrics()` returns every metric the spec reads.
- `group_of(idx)`, `features_in(groups) -> [idx]`, `as_mapping(vec) -> {name: value}`.

Cadence note: count, bytes, avg and clr columns are exactly cadence invariant. Ratio columns are not, because the +0.5/+1 smoothing depends on exposure n; the Beta-Binomial predictive in B04 is what handles exposure.

## detectors — registry (impl)

Constants:

| Name | Meaning |
|---|---|
| `DETECTORS` | 31 detector names; order is the store contract for `behavior.score`, `pm` and `p` |
| `N_DETECTORS` | 31 |
| `DETECTOR_INDEX` | `{name: idx}` |
| `FAMILIES` | 12 families |
| `FAMILY_DEFAULT_AXES` | default axes per family |
| `BUDGET_PATHS` | null alarms per entity-day, by decision path |
| `INSTANT_DETECTORS` / `ACC_DETECTORS` / `P2_DETECTORS` | detectors by kind or phase |
| `INSTANT_IDX` / `ACC_IDX` | indices of the instantaneous and accumulator detectors |
| `INSTANT_FAMILIES` | families that have instantaneous members |
| `FAMILY_MEMBERS` / `FAMILY_IDX` | members of each family, by name and by index |
| `CLASS_DETECTORS` | detectors scored at `class:<id>` |

`DETECTOR_INFO[name]` holds:
- `family`, `kind` (`'inst'` or `'acc'`), `axes` (the defaults), `owner` (e.g. `'B14'`) and `p2`;
- `strata` (`'daypart_cc'`, or `'daypart_regime'` for identity);
- `level` (`'entity'` or `'class'`);
- for accumulators only, `budget_path` and `budget_per_day` (the path budget split evenly over its detectors).

- `family_members(family, kind=None) -> [name]` returns a family's detectors in `DETECTORS` order.
- `family_of(d)`, `is_instant(d)`, and `detectors_of(owner) -> [name]` (the detectors an engine writes).
- `new_score_vector() -> float64[31]` returns an all-NaN row for score, pm or p.
- `arl_days(d)` returns 1/budget_per_day for an accumulator, the value to pass to `seq.h_for`.

## stages — kill-chain map (impl)

`STAGES` = behavior, off_hours, discovery, credential, privilege, collection, c2, exfiltration, identity, lateral. `FLAG_NAMES` lists the keyword flags below (all default False).

- `stage_for_axis(axis, *, sensitive, admin, external, upload_dominant, new_external_domain, low_prevalence, internal, object_ids, auth, login_4xx) -> stage | None` maps one axis to a stage (contract K).
  - Feature-group axes map to behavior.
  - categorical maps to privilege, exfiltration or c2 according to the flags.
  - breadth maps to collection (object ids) or otherwise discovery.
  - sequence maps to credential when the token family is auth or there are 4xx on login.
  - xsys maps to lateral.
- `stage_for_event(kind)`: identity events map to identity, and beacon maps to c2.
- `stage_for_category(lib4_category)`: recon/scan → discovery, auth/bruteforce → credential, tunnel/beacon/c2 → c2, transfer/exfil → exfiltration, admin → privilege.
- `stages_for(axes, flags) -> set` returns the distinct stages across a set of axes.

## classkeys — pseudo-entities (impl)

Constants: `SYSTEM_KEY='__system__'`, `ORG=('__org__','__org__')` (= (`ORG_SYSTEM`, `ORG_ENTITY`)), and the prefixes `CLASS_PREFIX`, `STATIC_PREFIX` and `POOL_PREFIX`.

- `role_key(rid) -> 'class:<rid>'` (keyed by id, never by name), `static_key(name) -> 'class:static:<name>'`, `pool_key(cidr) -> 'class:pool:<cidr>'`.
- `is_pseudo(entity)` is true for `__*` and `class:*`. `is_class(entity)`, `class_kind(entity) -> 'role'|'static'|'pool'|None`, `class_id(entity)`.
- `is_org(s, e)`, and `assign_key(s, ip) -> 'sys|ip'` (the key of `model.class.assign`).

## names — hosts (impl)

- `etld1(host)` returns the registrable domain using the public-suffix subset `PUBLIC_SUFFIXES_2L` (com.cn, gov.cn, edu.cn, org.cn, net.cn, co.uk, com.au, co.jp, …). IP literals and single labels come back unchanged.
- `split_host(host) -> (left_labels, etld1)` (used for DNS `{rnd}` templating). `normalize_host(host)` strips case, port, scheme and the trailing dot.
- `is_external(host, org_domains) -> bool` treats the following as internal: private or loopback IPs, single-label names, `INTERNAL_TLDS` (.local, .corp, …), and anything under an org domain.

## priors — hyperpriors (impl)

Dataclasses:
- `GammaPoissonPrior(mu0=1/min, a0=0.5)`, with `.b0 = a0/mu0` minutes;
- `BetaPrior(0.5, 0.5)`;
- `NIGPrior(m0, kappa0=0.01, alpha0=1, beta0=4)`, with `.df` and `.predictive_scale()`.

Constants:
- `KIND_M0` (bytes: ln 1e4), `FEATURE_M0` (per-feature m0 overrides), `COUNT_PRIOR` and `RATIO_PRIOR`;
- backoff constants: `BACKOFF_KAPPA_MIN/MAX` = 2 and 50, `RHYTHM_CLASS_STRENGTH` = 6, `DIRICHLET_BACKOFF` = 5, `NB_KAPPA_CLIP`, `BB_PHI_CLIP`.

Functions:
- `prior_for_kind(kind)` and `prior_for_feature(name)` return the org-tier hyperprior.
- `likelihood_family(kind)` returns `'nb'` (count), `'bb'` (ratio) or `'t'` (everything else).

Known gap: see `COLD_START_NOTE`. With a0 = 0.5, 10 req/min under the hyperprior gives two-sided p ≈ 3.4e-3, not > 0.05 (B03 test c).

## bayes — predictives (impl)

Tested in `tests/lib/test_bayes.py`. Dataclasses: `NIG(m, kappa, alpha, beta)` and `GammaRate(a, b)`, with `.mean`. All functions are vectorised with broadcasting, and NaN in gives NaN out. A scalar input returns a Python `float`. Scalar calls use a fast path through `scipy.special.cython_special`.

Constants: `PHI_CLIP = 1e-15`, `P_FLOOR = 1e-300`, `NB_KAPPA_CLIP = (0.5, 1e3)`, `BB_PHI_CLIP = (20, 1000)`, `NB_R_MIN = 1e-6`, `NB_R_POISSON = 1e12`, `BB_P_CLIP = 1e-6` and `BB_EXACT_MAX_N = 2048`.

Conventions:
- Every two-sided p (`*_midp`, `two_sided_midp`) is clipped to [`P_FLOOR`, 1].
- Observed counts k and n are rounded to the nearest integer in the pmf and mid-p functions, because they may carry float32 noise.
- `*_cdf` and `*_sf` use floor(k).
- An impossible observation (k < 0, k > n, or k = ±inf) gives NaN in the mid-p functions, never p = 0.

Normal and anchor helpers:
- `phi_inv(u, clip=1e-15)` returns the clipped normal quantile. `phi_sf(z)` returns the upper normal tail.
- `two_sided_midp(u)` returns min(1, 2·min(u, 1−u)).
- `combine_anchors(p_cur, p_ref)` returns min(1, 2·min(p_cur, p_ref)). If one of them is NaN, the other is used on its own, without the factor 2.

Negative binomial:
- `nb_pmf`, `nb_cdf` and `nb_sf(k, mean, r)`: NB(mean, size r) through `betainc`/`betaincc`. The incomplete beta never receives a rounded near-1 argument at large r or large mean (the upper tail also switches to `betaincc` on the directly formed small argument when mean > 1e4·r), and ln C(k+r−1, k) uses a Stirling difference. Together these keep pmf, cdf and sf exact to about 1e-12 relative for every r below the Poisson switch. For comparison, scipy.stats is off by 1e-8 at r = 1e6. sf = P(X > k) is computed directly.
- Parameter edges for these three functions:
  - r is clipped to [1e-6, 1e12], and r ≥ 1e12 is evaluated as Poisson.
  - mean ≤ 0 is a point mass at 0.
  - k < 0 gives cdf 0 and sf 1.
- A scalar `nb_cdf` call costs about 0.8 µs. `scipy.stats.nbinom.cdf` costs about 55 µs, so the ratio is at least 50× (asserted in the tests).
- `nb_mid_u(k, mean, r)` returns P(X<k) + ½P(X=k), evaluated as (F(k−1) + F(k))/2. `nb_midp(k, mean, r) -> (u, p2)`, where the upper side is (S(k−1) + S(k))/2. Both sides use only same-side tails, so there is no pmf term and no cancellation.
- `nb_size(kappa_hat, a_post)` returns 1/(1/κ̂ + 1/a), with κ̂ clipped to `NB_KAPPA_CLIP`. a_post ≤ 0 gives NaN.
- `nb_ppf(q, mean, r)` returns the smallest k with cdf ≥ q. For q > 0.5 it tests sf ≤ 1−q instead.
  - q ≤ 0 gives 0 and q = 1 gives inf. A q outside [0, 1] gives NaN.
  - The search starts at `nbdtrik`, which usually costs two tail evaluations. That is about 0.2 ms per call, or about 0.5 ms for 16 features × 3 quantiles in one call, so refresh `model_state` at refit cadence and not every tick.

Beta-binomial:
- `bb_logpmf(k, n, a, b)` uses rising-factorial (Stirling) differences lnΓ(x+d) − lnΓ(x), exact at any a + b (gammaln/betaln cancel at large a + b). A k outside [0, n] gives −inf.
- a + b overflowing to inf is invalid and gives NaN in `bb_cdf`, `bb_sf`, `bb_midp`, `bb_logpmf` and `bb_ppf` (never p = 1e-300). `bb_sf` is capped at 1.
- `bb_cdf` and `bb_sf` give each tail exactly, summed from k outward with the pmf ratio recurrence.
  - The sum covers up to `BB_EXACT_MAX_N` terms, so every n ≤ 2048 is exact.
  - Only the part of a tail more than 2048 counts beyond k comes from a Beta moment-matched to the jittered count (K+U)/(n+1). That Beta keeps the skewness.
  - Measured worst |log10 error| is 0.13 over n ≤ 1e5, mean in [1e-4, 0.999], c in [2, 1e6] and p in [1e-10, 0.5].
- **Contract change:** the previous "normal approximation with continuity correction above n = 200" was removed. For a 1 % ratio at n = 1000 and c = 50 it was off by more than 10 orders of magnitude, which would cause constant false alarms.
- `bb_midp(k, n, a, b) -> (u, p2)`. n ≤ 0 gives NaN. Cost is about 23 µs at n = 200 and 57 µs at n = 1000.
- Scalar `*_midp` calls return (NaN, NaN) when the predictive is NaN on either side (a NaN never becomes `P_FLOOR`). Per-feature loops in B04 should use the scalar path (about 2.3 µs per `nb_midp`); the array path has about 110 µs of numpy overhead for 16 features.
- `bb_params(p_hat, c) -> (a, b)`, with p̂ clipped to [1e-6, 1−1e-6]. c ≤ 0 gives NaN.
- `bb_ppf(q, n, a, b)`. Quantiles that share (n, a, b) reuse one pass.

Student-t:
- `nig_predictive(post) -> (df, loc, scale)`. An invalid posterior gives scale = NaN.
- `t_cdf(x, df, loc, scale)`. scale ≤ 0 gives NaN.
- `t_midp(x, post) -> (u, p2)`. Each tail comes from its own `stdtr`.
- `t_ppf(q, post)`. q = 0 gives −inf and q = 1 gives +inf (scipy's stdtrit(df, 0) = +inf bug is handled).

Posterior updates on sufficient statistics (all vectorised over feature arrays):
- `gamma_rate_posterior(a0, b0, Σx, Σe)`. Negative float residue in the sums is floored at 0.
- `count_overdispersion(W, Σx, Σe, Σx², Σxe, Σe²) -> κ̂`, clipped to [0.5, 1e3].
  - W ≤ 1, Σe ≤ 0, μ = 0 or excess ≤ 0 gives 1e3.
- `ratio_posterior(a0, b0, W, Σk, Σn, Σk²/n, Σ(k/n)²) -> (p̂, φ̂)`, with φ̂ clipped to [20, 1000].
  - W < 3, Σn ≤ 0, m ∈ {0, 1}, n̄ ≤ 1 or ρ ≤ 0 gives 1000.
  - Σ(k/n)² is accepted but not used.
- `nig_posterior(prior, W, Σx, Σx²)`. W ≤ 0 returns the prior.
- `nig_merge(a, b, prior, w_b)` inverts `nig_posterior` to recover the stats of a and b, then adds them. It is used for link seeding and pooling.

Note for B04 test (d): the mid-p of a discrete X is only approximately uniform. For Poisson(22) with 2000 draws, a KS test on it rejects at 0.01. KS-test the randomised PIT F(k−1) + V·P(X=k) instead, which is exactly uniform, or the continuous (t) features.

## combine — fusion primitives (impl)

- `whmp(ps, ws=None)` returns Σw/Σ(w/p) over the finite p. NaN is dropped, and all-NaN gives NaN.
- `randomized_conformal_p(ring_sorted, s, u)` returns (#{c>s} + u(#{c==s}+1))/(|C|+1). Trailing NaN in the ring are excluded and an all-NaN ring returns u. A float32 ring rounds s to float32 first (|s| beyond float32 range becomes ±inf, with no overflow warning).
- `e_day(p, dt_s)` returns p·86400/dt. `p_from_e_day(e, dt_s)` is its inverse. NaN dt gives NaN; dt ≤ 0 or infinite raises ValueError.
- `seeded_uniform(*keys) -> u ∈ (0,1)` is a deterministic blake2b uniform (8-byte digest read big-endian, (h + 0.5)/2⁶⁴).
- `e_day_severity(e, alpha_mult=1)` returns `'critical'|'high'|'medium'|'low'|None` from `SEVERITY_E_DAY` = critical 3e-6, high 3e-4, medium 3e-3, low 0.03 (e ≤ thr·alpha_mult, most severe first), before corroboration.
- `logit_blend(p1, p2, w1)` returns sigmoid(w1·logit p1 + (1−w1)·logit p2).

## calib — rings and tails (impl)

Constants: `RING_M=256`, `SMALL_N=64`, `TAIL_Q=0.90`, `MIN_EXCEED=10`, `GPD_REFIT_TICKS=16`, `KS_MAX_D`, `RATE_RATIO_BAND`, `XI_FLOOR=0.0`, `P_FLOOR=1e-300`.

Dataclasses:
- `GPDTail(u, xi, sigma, rate, n, fitted_ts=nan)`, with `.valid()`, `.sf(s)` (rate·gpd_sf(s−u), floored at 1e-300; s ≤ u gives rate; a NaN/None s or a tail that is not `valid()` gives NaN), `.to_dict()` and `.from_dict()`. The private scalar `_gpd_sf` returns NaN for invalid ξ/σ when y > 0, as `evt.gpd_sf` does.
- `Ring(cap, scores, ts, gpd)`: scores ascending (float32-rounded float64), with aligned ts. Construction normalises its inputs (rounds, drops non-finite, sorts, evicts down to cap).
  - `add(score, ts)` inserts in order and evicts the oldest ts (ties: lowest score first). Non-finite score or ts is ignored. O(log M) search plus an O(M) copy.
  - `remove_after(ts) -> n_removed` is used on rollback; it drops the GPD fit when anything is removed. NaN ts removes nothing.
  - `p_value(score, u)`, `quantile(q)`, `reset()`, `to_dict()` and `from_dict()`. to_dict is strict JSON (fitted_ts NaN is written as None).
  - Arrays are copy-on-write (never mutated in place), so a shallow copy or a held reference is a safe snapshot for checkpoints and replay.

Functions:
- `stratum_key(daypart, cc) -> 'wd_day|900'`, `identity_stratum_key(daypart, tercile) -> 'wd_day|r2'`, `ring_key(detector, stratum) -> 'd@stratum'`, `split_ring_key`. Malformed parts raise ValueError; cc and the tercile must be whole numbers (900 and 900.0 give the same key), and inf, NaN, None, strings and bools raise ValueError.
- `fit_tail(ring, now_ts=nan, xi_min=XI_FLOOR) -> GPDTail | None` fits a PWM-GPD (evt.gpd_pwm_fit) to the entries strictly above u = q_0.90, needs at least 10 of them, rate = N_u/N. A ξ below `xi_min` is raised to `xi_min` keeping the mean excess (σ = mean(y)·(1 − xi_min)). With the default of 0, a spuriously bounded fit becomes the exponential tail. `xi_min=-0.5` gives the raw fitter. Pure: the caller assigns `ring.gpd`.
- `p_from_ring(ring, s, u, tail=None)` uses `tail` (or `ring.gpd`) when s > tail.u: rate·gpd_sf, floored at 1e-300 and capped at 1/(n+1) when s is above every ring entry. Otherwise it returns the randomised conformal p (a private fast path, identical to `combine.randomized_conformal_p` for a ring that keeps its invariant; about 2.2 µs). s is float32-rounded first, and NaN gives NaN. O(log M).
- `blend_small_sample(p_conf, p_model, n, n0=64)` logit-blends with weight n/(n+64) through combine.logit_blend. n ≤ 0 gives p_model, and a NaN on either side passes the other through.
- `ks_uniform(ps) -> D` over the finite ps (NaN if none), and `health_weight(ks_d, rate_ratio) -> 0.5 | 1.0`, where a NaN input only neutralises its own check.
- `seed_ring(own, other, frac=0.5)` adds the most recent floor(own.cap·frac) entries of the linked entity's ring (link seeding), with oldest-first eviction. It returns a new Ring and leaves the inputs untouched.

Why the ξ floor and the cap: on a streaming 256-ring with a refit every 16 ticks, the spec-exact plug-in tail realised 4.7× nominal at p ≤ 3e-4 and 44× at 2e-5 on an Exp(1) null. It sometimes gave p = 1e-300 to null ticks past a spurious end point. With the floor this becomes 1.5× and 2.5×. Heavy-tailed nulls (ξ > 0) are unaffected and stay about 3× anti-conservative at 3e-4, from ξ sampling noise with about 26 exceedances.

## evt — extremes and point processes (impl)

Constants: `XI_CLIP=(-0.5, 0.5)`, `MIN_INTERVAL_S=1e-3`, `KAPPA_MAX=1e6`, `Z2_MAX_TRIALS=512`, `Z2_MIN_OVERSAMPLE=2`, `P_FLOOR=1e-300`.

GPD and peaks over threshold:
- `gpd_pwm_fit(exceedances) -> (xi, sigma)` uses the Hosking–Wallis PWM estimators, with ξ clipped to [−0.5, 0.5] (σ = mean·(1 − ξ) when clipped). Non-finite values are dropped; n < 3 or mean ≤ 0 gives (0, max(mean, 1e-12)). It recovers ξ within 0.1 on 2000 samples for ξ ∈ [−0.3, 0.4].
- `gpd_sf(y, xi, sigma)` is vectorised and keeps y's shape: y ≤ 0 gives 1, beyond the ξ < 0 end point gives 0, NaN gives NaN, and invalid ξ/σ gives NaN wherever y > 0. It equals calib's scalar `_gpd_sf` to 1e-12.
- `pot_sf(x, u, xi, sigma, rate)` returns rate·gpd_sf(x − u) for x > u, and **1.0 for x ≤ u** (the body carries no tail evidence; use the empirical p there). rate is clipped to [0, 1].
- `pot_quantile(u, xi, sigma, rate, q)` returns the level z_q with P(X > z_q) = q, via expm1. rate is clipped to [0, 1] as in `pot_sf`, so the two stay inverse. q ≥ rate gives u; q → 0 gives +inf, or the end point u − σ/ξ when ξ < 0; overflow gives +inf instead of raising. σ ≤ 0 or non-finite, or any NaN input, gives NaN.

Gamma renewal test for beacons:
- `gamma_renewal_lrt(intervals) -> (kappa_hat, LR)` initialises with Minka's approximation, refines by up to 4 Newton steps, and is one-sided (κ̂ > 1). It matches scipy's Gamma MLE and log-likelihood ratio to 1e-9.
  - Non-finite intervals are dropped and n counts the finite ones. n < 3 gives NaN. κ̂ is capped at 1e6: equal intervals give LR ≈ 14n. Costs about 25 µs.
  - s = ln mean − mean ln is computed without cancellation, and for κ ≥ 20 digamma, trigamma and lnΓ use asymptotic series.
- `beacon_null_p(LR, n)`: **n counts intervals, that is events − 1.** It uses the row of the largest tabulated n' ≤ n (conservative, because rows decrease in n). n < 8 gives NaN. log10 p is interpolated linearly in LR, which is conservative by at most 0.025 decades and never anti-conservative. Below the first knot the anchor is (0, p = 1). Beyond the last knot it extrapolates log-linearly with the exact asymptotic slope of the row, p = p_last·exp(−(n'−1)/(2n')·(LR − LR_last)), floored at 1e-300; ln p is convex in LR, so this line is an upper bound (within about 0.06 decades of the exact inversion). LR ≤ 0 gives 1. Costs about 1 µs.
- **Contract change (table):** `BEACON_NULL_N = (8, 9, 10, 11, 12, 14, 16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256)` (was 12..256, which gave NaN for a 12-event stream) and `BEACON_NULL_LOG10P` = log10 0.5, then −0.5 to −12 in steps of 0.5 (was to −5). Alarm-level p is now interpolated, not extrapolated.
- The table is the **exact** null, not a Monte-Carlo one. The LR is monotone in s, and T = n·s has the closed-form Mellin transform n^(−nu) Γ(n) Γ(1−u)^n / Γ(n(1−u)), which is inverted numerically on a Bromwich contour.
  - `scripts/gen_beacon_null.py` regenerates the table and checks it against 1e5 MC draws per n (max |z| = 2.6 over 150 cells) and against importance sampling at 1e-6, 1e-9 and 1e-12 (within ≈0.01 decades).
  - Measured: a Poisson null with n ∈ {12, 20, 40} gives at most 0.4 % of p < 1e-3 over 1000 streams. ±30 % jitter at P = 300 s gives median p = 7.6e-8 at 12 intervals, and 2.7e-7 at 11 intervals (12 events).
- `build_beacon_null_table(...)` is the offline MC estimator, used as a cross-check. Columns with n_sims·p < 10 are NaN.

Periodicity:
- `z2_trial_periods(span_s, p_min_s=10, p_max_s=None, oversample=5) -> (periods, n_eff)` returns periods descending, uniform in frequency, with n_eff = (f_max − f_min)·span over the scanned grid.
  - Capped at 512 trials: the step first coarsens, down to 2× oversampling, and after that the grid keeps the long-period end (about 255 Fourier frequencies above 1/p_max).
  - Degenerate inputs give (empty, 0).
- `z2_periodogram(times, trial_periods, m=2) -> Z²[]` is shift-invariant. NaN times are dropped; n < 2 gives NaN, and invalid periods give NaN at their positions.
  - **Contract change:** it is window-corrected and conditions on the two endpoint events, which define the window: Z² = (2/(n−2)) Σ_k |Σ_i e^(ikφ_i) − (1 + e^(iθ_k)) − (n−2)·w_k|², with θ_k = 2πkT/P and w_k the phasor mean of uniform times over [t_min, t_max]. The uncorrected Z² was 12× anti-conservative at n = 256, and subtracting n·w_k alone was still 1.25–1.5× anti-conservative at 1e-3. A strict train over whole cycles gives 2m(n−2); n = 2 gives 0.
  - A uniform grid takes a factorised path (about 0.45 ms at 256 × 512); any other grid uses the direct O(nL) sum. More than about 3 Z² evaluations per tick exceed a 2 ms budget, so B12 should round-robin them or skip Z² when the renewal p already alarms.
- `z2_p(z2_max, m, n_eff, n_trials=None)`: **Contract change:** it uses the Davies upcrossing bound, χ²_2m.sf(z) + n_eff·√(π(m+1)(2m+1)/18)·(z/2)^(m−½)e^(−z/2)/Γ(m), floored at 1e-300.
  - The previous n_eff·χ².sf(z) was measured at 2–5× anti-conservative on the oversampled scan.
  - The Davies bound alone realises 0.2–0.9× nominal.
  - The new optional `n_trials` is len(trial_periods). When it is given, the union bound n_trials·χ².sf(z) also applies and the smaller of the two is used, which measures at most about 1.0× (0.15–1.0×) at 1e-1..1e-3 for n = 12..256 over 2e4 Poisson streams per case. **B12 should pass it.**
  - Z²₂ ≤ 4(n−2) caps what 12 events can reach (p ≈ 2e-5); below that, beacons are the renewal LRT's job. B12 event times need sub-second resolution: equal tick timestamps are clipped to 1 ms intervals, which is conservative but blinds the renewal test.
- `rayleigh_p(times, period)`, with Zar's correction, shift-invariant. n < 2 or an invalid period gives NaN.

## seq — sequential statistics (impl)

Thresholds:
- `arl_ticks(arl_days, dt_s)` returns arl_days·86400/dt.
- `h_gauss(k, arl_ticks)`: Siegmund threshold for N(0,1) inputs, e.g. 19.4 / 5.35 at 900 s for ARL 2400 d.
- `h_evidence(arl_ticks)` returns (ln ARL − 3.07)/0.94. `h_llr(arl_ticks)` returns ln ARL.
- `bernoulli_threshold(arl_slots)` returns log2 ARL + 0.5.
- `mcusum_h(d, arl_ticks)` interpolates the filled `MCUSUM_H` table (16 rows d = 1..16 × `MCUSUM_ARL_GRID` = 1e2..1e7) linearly in ln ARL, extrapolating linearly beyond. d > 16 uses row 16 (too low a threshold: B14 must keep d ≤ 16); d < 1 uses row 1.
- `h_for(kind, arl_days, dt_s, k=0.5, d=1)` is the dispatcher over `'gauss'|'evidence'|'llr'|'bernoulli'|'mcusum'`; an unknown kind raises ValueError. (architecture.md's `seq.h_for(kind, k, ARL_days·86400/Δt)` row predates this signature; this one is authoritative.)
- ARL ≤ 1 gives 0 (`bernoulli_threshold` counts it as 1, giving 0.5) and NaN gives NaN. For an infinite ARL, `h_gauss` returns `H_MAX` = 200 while the others return inf.

Steps:
- `cusum_step(S, x, k)` computes max(0, S+x−k). A NaN x leaves S unchanged.
- **NaN state:** a NaN chart state restarts at 0 in `cusum_step`, `mcusum_step` (the whole vector of that row), `evidence_cusum_step` and `bernoulli_cusum_step`, and the current increment still counts. Without this a stored state that round-tripped a JSON null would silence the chart forever.
- `cusum_stationary_p(S, k, n_charts=1)` returns min(1, n·exp(−2k(S+0.583))).
- `evidence_cusum_step(S, q_inst)` computes max(0, S − ln q − 3).
- `rhythm_p1(p0)` returns min(.95, max(.5, 5p0)); it is NaN for p0 > 0.3.
- `bernoulli_cusum_step(W, a, p0, p1)` updates W in bits.
- `mcusum_step(S, x, k=0.5) -> (S', stat)` is the Crosier MCUSUM (NaN x entries count as 0). It works on the last axis, so a batch [..., d] returns an ndarray stat.

Prewhitening and trend:
- `ar1_phi(x)` returns φ clipped to [0, 0.8], NaN-aware.
- `prewhiten(x_t, x_prev, phi)` returns (x−φx_prev)/√(1−φ²); a NaN x_prev gives x_t and a NaN φ means no whitening. Callers clip to ±`PSI_CLIP` = 3.
- `mann_kendall(x) -> (S, p)` (tie-corrected, continuity-corrected; n < 4 gives (0, 1)), `sen_slope(x, t=None)` (no usable pair gives NaN; len(t) ≠ len(x) raises ValueError).

## gating — reversible learning (impl, contract H)

Constants:
- `D_MIN_S=600`, `D_MIN_TICKS=4`;
- `CKPT_EVERY_S=3600`, `CKPT_EVERY_REF_S=86400`, `CKPT_AGES_H`;
- `JOURNAL_MAX_AGE_S` = `HELD_MAX_AGE_S` = 8 d;
- `ROLLBACK_MAX_DEPTH_S` = 7 d, `ROLLBACK_MIN_INTERVAL_S` = 1 h;
- `LINK_SEED_WEIGHT=0.5`, `DEFAULT_CLOCK='feature.active'`.

Frontier and candidates:
- `commit_delay_ticks(dt_s, d_min_s=600)` returns max(4, ceil(D_min/dt)). `commit_frontier(now, dt_s, d_min_s)` returns now − D·dt.
- `commit_candidates(store, s, e, learner, last_ts, now, dt_s, *, d_min_s, clock, training) -> [(ts, w_eff, w_prov)]` lists every clock tick in (last_ts, frontier]. A missing trust value gives 1 in training and 0 live.
- `is_quarantined(store, s, e, now, dt_s)` returns the latest non-NaN `behavior.quarantine` before now (NaN counts as missing; none gives False). A NaN from a degraded B28 tick never opens the gate.

Dataclasses:
- `CommitRow(ts, w_eff, w_prov)`: an immutable NamedTuple (changed from the stub's slots dataclass; same fields and constructor) so `to_dict()`/`from_dict()` share rows instead of rebuilding an 8-day journal every tick. A released row is journaled with `w_eff = w_prov`.
- `GateState(last_ts, version, branch, held, journal, last_ckpt_ts, applied, frozen, allow_drift, link_version, last_rollback_ts)`, with `.to_dict()` (JSON-safe) and `.from_dict()` (a GateState passes through). It is persisted inside the learner's own model. Checkpoint-validity bookkeeping (barriers, rebase hooks, link seeds, last-rollback diagnostics) lives under `_`-prefixed keys of `applied`.

`GatedLearner(name, init, update, fetch, dump, load, merge=None, on_rebase=None, d_min_s, ckpt_every_s, clock)`:
- `.step(store, s, e, state, gate, now, dt_s, training) -> (state, gate)`: applies control, then commits or holds each candidate, then checkpoints at most every `ckpt_every_s`.
- `.apply_control(store, s, e, state, gate, control, now, dt_s)` handles each directive once:
  - rollback_to restores the checkpoint, replays the journal rows with their recorded w_eff, and moves later rows to held;
  - release commits the held rows in [t0, t1] with w_prov;
  - rebase_from commits the held rows with w_prov into version+1 (calling `on_rebase`);
  - frozen clears held and stops commits;
  - allow_drift is recorded.
- `.seed_from_link(store, s, e, state, gate, load_other)` computes B := B_own + 0.5·A when `model.link.version` increases.

Other functions:
- `control_directives(control)` normalises `model.control`.
- `reference_eligible(ts, w_eff, incident_or_regime_ts, window_s=86400)` is the reference-anchor admission rule.

Store calls: `vec_since`, `vec_at`, `get_model` (`model.control` and `model.link`), `get_checkpoint` and `put_checkpoint`. It never calls `put_model`; the engine persists the state and the GateState.

Rollback coverage: store checkpoint retention keeps a floor past 168 h, the EARLIEST checkpoint of every absolute day inside the horizon (never thinned), and up to 7 geometrically thinned checkpoints in the newest 24 h (max 16 per key). Any rollback target up to 168 h back therefore has a checkpoint within 24 h before it, inside the 192 h journal, so replays are exact (`applied['_last_rollback']['complete']` is True); `complete = False` remains as a guard. B28 governor must also set `behavior.quarantine` = 1 at or before the tick where it writes `rollback_to`, and 0 from the tick it writes `release`.

## timebins — local time (impl)

Constants: `DEFAULT_TZ='Asia/Shanghai'`, `DEFAULT_DAY_HOURS=(8,20)`, `SLOT_S=900`, `CADENCE_CLASSES=(60,300,900,3600)`, `DAYPARTS`, `DAY_TYPES`, `TCTX_FIELDS`.

- `Calendar(holidays, makeup_workdays)` normalises both to `frozenset[date]` (ISO 'YYYY-MM-DD' strings, dates, datetimes, lists and sets are accepted; a bad date raises ValueError), and `parse_calendar(cfg)` builds one from config (cached).
- `day_type(date, calendar) -> 'workday'|'nonworkday'` handles holidays and 调休. `day_type` and `tctx` also accept the raw `ctx.config['calendar']` mapping in place of a Calendar.
- `cadence_class(dt_s)` returns the nearest class in log space.
- `local_datetime(ts, tz)`, `slot_of(ts, tz)`, `slot_bounds(slot, tz) -> (t0, t1)` (DST-aware).
- `daypart(day_type, hour_local, day_hours) -> 'wd_day'|'wd_night'|'nwd_day'|'nwd_night'`.
- `tctx(ts, tz, calendar, day_hours, dt) -> {hour_local, dow, day_type, bin48, bin168, slot, daypart, cc}`, and `tctx_from_config(ts, config, dt)`.
- `encode_tctx(dict) -> float32[8]` and `decode_tctx(vec) -> dict` convert to and from the vec-ring encoding.
- `hour_distance(h1, h2)` returns the circular difference in (−12, 12]. `von_mises_weights(hour, kappa=4, max_dh=2) -> {bin: w}`.
- Tested in `tests/lib/test_timebins.py`. All tctx fields come from one local clock, L = ts + utcoffset(ts), so hour, bins and slot always agree at a boundary. bin48 = hour + 24·nonworkday. slot counts 15-minute slots since the local epoch, not slots of the day.
- DST: `slot_bounds` of a slot skipped at spring-forward is empty at the transition instant. A slot repeated at fall-back returns its first pass. A slot only partly skipped (a jump not aligned to 15 minutes) starts at the transition.
- `parse_calendar` and `tctx_from_config` raise `ValueError` on a bad date, tz or `daypart_day_hours`. `daypart_day_hours` accepts `[8, 20]` or `"8-20"`, and a start > end wraps midnight. A non-finite ts or hour raises `ValueError`. For `cadence_class`, NaN raises, dt ≤ 0 gives 60 and +inf gives 3600. `decode_tctx` returns non-finite int fields as None, except hour_local, which stays NaN, and raises ValueError on a day_type or daypart index that is not an in-range integer. `encode_tctx` rounds hour_local down to the last float32 inside the same hour, so a stored row never contradicts its bin48.

## sketch — hashing sketch and HLL (impl)

Constants: `SKETCH_NAMESPACES` (act.tokens, client.stack_set, sni_etld1, dns_etld1, l4.dport_set), `SKETCH_BLOCK=16`, `SKETCH_DIM=80`, `HLL_P=10`, `HLL_M=1024`.

- `token_hash(ns, tok) -> (bucket, sign)` uses blake2b of `f'{ns}\x1f{tok}'` (UTF-8, 8-byte digest **read big-endian**): bucket = x % 16, sign = bit 32. A non-str tok is hashed as str(tok), so the port 443 and '443' are the same token. Memoised per namespace; ≈50 ns warm, ≈1.3 µs cold.
- `sketch_block(counts, ns) -> float64[16]` returns a unit-norm block (1 ± 1e-9, never NaN; an exact opposite-sign cancellation falls back to unsigned), or zeros when the namespace is empty. '__other__' and counts that are ≤ 0, non-finite or non-numeric are ignored; text counts (str, bytes, bytearray, also inside `{'n': ...}`) are non-numeric, so "3" is not a count. It is scale-free (15× the counts or 1e300-sized counts give the same block).
  - Accepted value shapes: `{tok: n}`; `{tok: {'n': …}}` (client.stack_set as stored); a set or list of tokens (each occurrence counts 1); None (empty).
- `sketch_vector(ns_counts) -> float32[80]`. Unknown namespaces are ignored. **Because of the float32 cast, block norms are 1 ± ~1e-7, not 1e-9**: the 1e-9 check belongs on `sketch_block`. A heavy entity (256 act.tokens + 144 others) costs ≈0.16 ms warm.
- `HyperLogLog()`, with `.add`, `.add_many`, `.count()`, `.merge(other)`, `.to_bytes()` (1024 B) and `.from_bytes`.
  - Items are hashed as str(item). `add_many` gives the same registers as repeated `add`, with the register update vectorised (≈0.07 s per 1e5 items).
  - `merge` is in-place, returns self and is lossless (the merged sketch equals the sketch of the union). `from_bytes` copies, and raises ValueError unless it gets exactly 1024 bytes with every register ≤ 55. Equality compares registers.
  - Accuracy is the textbook p = 10 figure, sd ≈ 3.25 %. Measured over 150 sets of n = 2000: rms 3.4 %, 86 % within 5 %. At n = 1e5 the rms is 3.1 %. `range(2000)` gives −1.3 % and `range(100000)` gives +0.1 %. **A single arbitrary id set misses a 5 % bound about 1 time in 8**, so engine tests should use a fixed set or a ≥ 4σ (13 %) bound. For example, ids 1000..2999 give −6.8 %.

## ppm — sequence model (impl)

- `PPMModel(order=3, half_life_s=30 d, counts, t_ref, g, n_tokens, stats, n_obs)`, with `.to_dict()` (JSON-safe deep copy; counts at stored scale, so a round trip is exact; tuple / numpy-scalar tokens are encoded, `half_life_s=inf` is written as None) and `.from_dict()` (None / {} gives a fresh model; a PPMModel passes through).
  - **Additive layout change:** `n_obs` = undecayed observation count per order-0 symbol (Good–Turing singletons are `n_obs == 1`, so N1/N is invariant to a uniform weight such as stream_frac ×4). Private caches `_tot` (stored total per context) and `_n1` (stored singleton mass) are rebuilt automatically if `counts` is replaced; mutate counts only through `update` / `merge`.
- `update(model, tokens, w=1, ts=0, history=()) -> model` is a lazy-decayed update-all, in place, ≈1.5 µs/token at order 3.
  - The clock is the newest ts seen. An older row is added decayed (w·2^(−Δt/H)), never the state backwards, so any row order gives the same counts (GatedLearner release / replay). An empty model anchors `t_ref` at its first update. g > 32 renormalises and drops entries whose unscaled count is < 1e-9.
  - w ≤ 0 or non-finite is a no-op. A NaN ts means "at the clock". `history` supplies the preceding context only (its tokens are not counted).
- `loglik(model, tokens, backoff=(...), vocab_size=0, history=()) -> bits[len]` walks the chain entity → class → system unigram, with Good–Turing novel mass at the last tier. Read-only, never inf, ≈1 µs/token on the predicted path.
  - Exclusion is inside each tier (both n and q drop excluded symbols), so one tier is exactly normalised over the vocabulary; a chain is sub-normalised (novel-to-entity symbols are charged slightly more, never less).
  - `None` and empty tiers are skipped; the last NON-empty tier does the Good–Turing step, with g clipped to [0.5/(N+1), 1 − 0.5/(N+1)]. An all-empty chain scores log2(max(1, vocab_size)) per token.
- `entropy_rate(model) -> (mu, sigma)` in bits/token, (NaN, NaN) when W < 1. `record_surprisal(model, surprisal, w)` folds finite entries into `stats`. The stats decay with the counts' half-life whenever the clock advances, so early immature-model surprisal does not pin mu high.
- `merge(own, other, w=0.5)` is in place on `own` (returned). `other` is folded like an update at its own clock (so it arrives decayed if older, or advances own's clock if newer); contexts longer than `own.order` are dropped; `n_obs` and `stats` merge with the same weight.
- `good_turing_unseen(model)` is the raw N1/N in [0, 1] (1.0 for an empty model). `vocab(model)` gives the unscaled order-0 counts.
- Clock edge cases: a NaN ts on an **empty** model leaves `t_ref` = NaN (unanchored; `to_dict` writes None and `from_dict` reads it back as NaN) and the first finite ts anchors the clock. A no-decay model (`half_life_s=inf`) keeps `t_ref` at its newest ts. `from_dict` renormalises a stored g above `PPM_RENORM_G`. Always pass a real ts: the default ts=0 followed by a real timestamp decays the earlier rows away.
- `merge` does not create entries whose unscaled mass would be below 1e-9 (the same threshold as `update`), and adds `n_obs` only for symbols that end up in the order-0 counts.

## robustcov — covariance and contributions (impl)

Constants: `WINSOR=4`, `C_STEP_H=0.75`, `EIG_FLOOR_REL=1e-3`, `PCA_VAR_FRAC=0.9`, `CHOL_CACHE_SIZE=8`. Every p is clipped to [1e-300, 1]; NaN in gives NaN out.

Fitting:
- `winsorize(X, c=4)` clips to [−c, c]; NaN is kept and ±inf become ±c.
- `oas(X, w=None) -> (mu, Sigma, rho)` is Chen et al. eq. 23 with n_eff = (Σw)²/Σw² (the docstring formula, which keeps the 2/p terms that sklearn drops).
  - Rows with any non-finite value, and rows whose weight is not finite and > 0, are dropped. Equal weights give exactly the unweighted fit.
  - n < 2 gives (mean, I, 1.0), with mean = zeros(p) when no row is usable. A non-positive denominator (S ∝ I, or p = 1) gives rho = 1.
  - **Caveat (measured):** rho reflects the whole spectrum, so one strong pair among many independent dimensions is shrunk hard. A ρ = 0.95 pair fitted at n = 336 keeps a regression slope of 0.94 at p = 2, 0.90 at p = 5, 0.54 at p = 20 and 0.16 at p = 52.
- `c_step_oas(X, w=None, h=0.75) -> (mu, Sigma)`. The signature is unchanged; three **contract changes (maths)**, all measured in the module docstring:
  - **Start:** the DetMCD spatial-sign start, refined by an OAS fit of its ⌈n/2⌉ innermost rows. It replaces the OAS of all winsorised rows, which a tight 20 % cluster masks (p = 5: KL 1.17 against 0.004 clean). With 20 % contamination the fit's KL(fit ‖ truth) is 0.003 for a tight cluster, 0.004 for 5-sd scatter, 0.005 for 3 dims shifted 8 sd, and 0.025 for a (+3, −3) pair break.
  - **Up to 3 C-steps**, stopping when the h-subset repeats (was one).
  - **Leave-one-out d² for the h-subset rows** (closed form, Sherman–Morrison, ρ and μ held fixed), so every d² used in the median rescale and the 0.975 reweighting is out-of-sample. In-sample subset distances are about 1.27× too small at 252 × 52, which cut about 21 % of clean rows and gave 2.0–3.1 % Hotelling false alarms at the 1 % level. Now 4–5 % of clean rows are cut and false alarms are 0.7–0.9 % at 1 % (336 × 52, factor correlation).
  - **Consistency rescale after the reweight:** the median d² of the inliers is matched to χ²_p.ppf(0.4875), the median of χ²_p truncated at 0.975. The stub had no correction, and a truncated fit is biased low. Hotelling KS D is 0.014–0.039 over 2000 fresh n = 200 fits for p ≤ 20; the stub's pipeline gave up to 0.062. The price is that a strongly anisotropic Σ comes out about 10 % low along its major axis.
  - Otherwise as specified: the median(d²)/χ²_p.ppf(0.5) rescale of the h-subset fit (d² recomputed under that fit), the reweight at χ²_p.ppf(0.975), OAS again, and eigen_floor. Weights enter every OAS and both medians, but not the start.
  - Fewer than 2 usable rows gives oas's (mean, I). Costs about 2.8 ms at 336 × 52 and 6.4 ms at 600 × 64 (one thread).
  - **Known limit:** ties or point masses (zero-inflated features, mid-PIT scores of low-mean counts) make any MCD-style fit collapse onto the tie (14–23 % false alarms at 1 %). B04 should emit z from a seeded randomised PIT.
- `eigen_floor(Sigma, rel=1e-3)` symmetrises. A matrix already above the floor is returned untouched; one whose mean eigenvalue is not > 0 is floored at rel.
- `pca_k(Sigma, var_frac=0.9) -> (U_k, lam_k, k)`: eigenvalues descend, and each column's largest |entry| is positive (a deterministic sign). 1 ≤ k ≤ p, and the fraction is compared with a 1e-12 tolerance, so I₁₀ gives k = 9.

Scoring:
- `hotelling_pred_p(t2, n, q)` uses the F-scaled prediction distribution; n may be fractional (n_eff). NaN t2, q < 1, or n ≤ q + 1 give NaN, and t2 ≤ 0 gives 1. It is exact for the sample mean and unbiased covariance (KS D < 0.05 at q = 20, n = 200, where χ²_q is anti-conservative).
- `wilson_hilferty(t2, q)`: NaN t2 or q < 1 gives NaN.
- `spe(z, U_k)` forms the residual explicitly; NaN in z gives NaN. U_k with k ≥ p gives exactly 0, so the Box parameters and `spe_p` are NaN (no SPE evidence).
- `spe_box_params(spe_train) -> (g, h)` uses the ddof = 1 variance over finite values. Fewer than 10 values, var = 0 or mean ≤ 0 give (NaN, NaN).
- `spe_p(spe, g, h)`: NaN or non-positive parameters give NaN.
- `rbc_contributions(z, Sinv) -> (rbc, p)`, each term χ²₁. With missing (non-finite) z_f, the observed terms use the observed sub-model's precision, inv(Σ_oo) = Sinv_oo − Sinv_om Sinv_mm⁻¹ Sinv_mo, so they stay χ²₁; missing terms are NaN.
- `conditional_impute(z, Sigma, observed)` returns a new array. Entries marked observed but non-finite are treated as missing. The completion keeps T² equal to that of z_o on Σ_oo, and (Σ⁻¹z)_m = 0.
- `CholCache(Sigma, maxsize=8)`, with `.set_sigma`, `.factor(observed) -> (idx_o, L)` and `.t2(z) -> (T2, q)`.
  - Sigma is copied and symmetrised, and cached (idx_o, L) arrays are read-only. A cache hit returns the same tuple object. maxsize ≤ 0 disables caching.
  - t2 uses the finite entries of z (±inf counts as missing). Costs about 6 µs warm and about 25 µs for a new pattern at p = 52.
- **Engine notes (B06):** pass n = the number of training rows to `hotelling_pred_p`. Build `Sinv` for RBC from the same Σ that CholCache holds. Centre before scoring: `CholCache.t2`, `conditional_impute`, `rbc_contributions` and `spe` expect z − μ with μ from `c_step_oas`. Fit the SPE Box parameters on out-of-sample SPE (held-out rows or the previous model), not the training rows (in-sample Box gives 2.9–4.0 % SPE false alarms at 1 %). **Spec issue:** engines.md B06 test (a) expects (+2, +2) at ρ = 0.95 to give T² p < 0.05, but T² = 8/1.95 = 4.1 on the true Σ (χ²₂ p = 0.129). (+3, +3) gives p ≈ 0.01.

## template — Drain-lite and token formats (impl)

Token formats, exact (see the module docstring for the full rules):

| Channel | Format | Example |
|---|---|---|
| HTTP | `'{METHOD} {host} {template}\|{2xx}'` | `GET erp.corp /orders/view/{num}\|2xx`, `/search?page&q` |
| TLS | `'{etld1}:{dport} u{a}/d{b}'` | |
| DNS | `'{QTYPE} {templated_qname}'` | |
| L4 | `'{proto}/{service} u{a}/d{b}'` | |
| eviction | `'{rare}'` | |

Constants: `MASK_TOKENS`, `OBJ_MASKS`, `VAR_TOKEN='{var}'`, `RND_TOKEN='{rnd}'`, `RARE_TOKEN='{rare}'`, `VOCAB_CAP=4000`, `MAX_DEPTH=6`, `SERVICE_PORTS`, `HTTP_METHODS`, `WRITE_METHODS`, `DNS_QTYPES`.

Masking and token builders:
- `mask_segment(seg) -> (masked, value|None)`, `mask_path(path) -> (segments, param_names, masked_values)`.
- `status_class(status)`, `size_class(up, down)`.
- `dns_label_is_random(label)`, `dns_template(qname)`.
- `tls_token(...)`, `dns_token(...)`, `l4_token(...)`.
- `channel_of(token)` and `token_family(token)` (for example `'http|read|erp.corp|orders'`).

`Templater()` is persisted as `model.template@(s, __system__)`. Its methods are `.template_path(host, method, path, w=1) -> (template, masked_values)`, `.http_token(method, host, path, status, w=1) -> (token, masked_values)`, `.intern(token, w=1) -> id`, `.token_of(id)`, `.maintain()`, `.to_dict()` and `.from_dict()`.
- A new literal path segment reads as `{var}` for its first 2 hits (provisional), so a brand-new endpoint shows up under its own literal template only from its 3rd hit (matters for R2 and B08 novelty). Consecutive `{rnd}` DNS labels collapse into one.
- Paths deeper than 6 segments end in `/a/b/c/d/e/f/{var}`. Cost is about 1.5–4.7 µs per token (mixed-stream average about 2.6 µs).

## stack — client stack tokens (impl)

Token format: `'ja3n|ua_family/major|os|ttl_class|win_class'`, for example `…|python-requests/2|linux|64|w15`.

Constants: `TTL_CLASSES`, `OS_FROM_TTL`, `WIN_CLASS_RANGE`.

- `ja3n(ja3)` drops GREASE (`(v & 0x0f0f) == 0x0a0a`) from ciphers, extensions and curves, sorts the extensions numerically, and returns the md5. It accepts 3–5 field JA3 strings; a bare 32-hex hash is passed through lower-cased (so real per-connection Chrome 110+ JA3 hashes would each give a new token); malformed input gives '-'.
- `ua_parse(ua) -> (family, major, declared_os|None)`. Unlisted browsers give family 'other'. `major` is always canonical ASCII decimal (Unicode digits are converted; 10⁹ or more gives '0'). The fallback parser is linear time (worst about 150 µs for a 512-character UA).
- `ttl_class(ttl)`, `win_class(win)`.
- `stack_token(ja3, ua, ttl, win_size, ja4=None)`. The os field is the UA-declared OS if there is one, otherwise the TTL-implied OS.
- `stack_id(token) -> int31`, the id shared by R2 and R3: blake2b-8 read big-endian, masked to 31 bits, with 0 mapped to 1. `parse_stack_token(token) -> dict` is the exact inverse of `stack_token` and never raises.
- `os_ttl_consistent(declared_os, ttl_cls) -> bool|None`: None for TTL class 255 (a network device re-sent the packets), a missing or unknown declared OS; class 32 is consistent with every OS.

## replay — deterministic recompute (stub, later wave)

- `replay(step, state0, inputs, perturb=None, threshold=None) -> ReplayResult(ts, scores, final_state, alarmed_at)`.
- `counterfactual_features(step, state0, inputs, candidates, threshold, neutral) -> [idx]`.
- `load_inputs(store, s, e, names, since, until)`.

## featcache — per-tick read cache (stub, optional)

- `FeatCache(store, now)`, with `.reset(now)`, `.vec(s, e, name)` (fresh-only and read-only) and `.feature(s, e, feature, which='vec')`.
