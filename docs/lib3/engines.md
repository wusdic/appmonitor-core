# Behavior Library v2.1 — engine specifications
Each engine: purpose, algorithm, store reads/writes, perf budget and the unit test it must pass.

**Status (docs sync after evaluation round 3, 2026-09-28).** Every engine below is implemented and registered in `backend/app/pipeline/build.py::build_registry()` except the P2 engines B19–B22 (specified, not built, not registered). Each section is the design; the "Integration notes (as built)" and "Canonical grain mode" paragraphs record where the code differs or was corrected, with the integration.md section that has the evidence. Canonical grain mode (`grain_mode='canonical'`, docs/lib3/cadence.md) is the default for the pipeline, the Runtime, eval and scripts; engine unit tests run the v2 'tick' mode, which the tick-mode golden test (`tests/test_tick_mode_golden.py`) pins to v2. Every engine runs at registry interval 1; the strides quoted in the headers are internal (`Engine.period_s`, `entity_due` or a tick counter). Latest gate results: integration.md §10.4 and §11. References to `anomaly.py`, `fingerprint.py`, `clustering.py`, `drift.py` and line numbers of v1 files name the deleted v1 engines each design replaced (integration.md §2).

## R1 — L4FlowEngine/HTTPEngine/TLSEngine/DNSEngine/L2L3Engine [upgrade]

- **File:** `backend/app/engines/raw/{l4flow,http,tls,dns,l2l3}.py`
- **Layer / order / interval:** raw / 1 / 1

**Purpose.** Deliver complete, weighted and fresh categorical and count evidence to library 3. Stop losing mass to top-k truncation, accept aggregated flow records, and keep pseudo-entities out of the entity registry.

**Algorithm.**

1) Weighted records. Let w = int(obs.extra.get('count', 1)). Every counter and sum adds w. Byte totals use obs.extra['bytes_up_total'] / ['bytes_down_total'] when present, otherwise w·bytes, and retransmits use obs.extra['retransmits_total'] when present, otherwise w·retransmits (every per-record field of an aggregate is a per-flow value; eval round 2: the generator used to put the group's retransmit total in the per-flow field, so l4.retransmit_rate of aggregated ticks was ~w× too high). Averages are w-weighted. This supports NetFlow/IPFIX-style aggregates and the generator's aggregated mode.
2) Full sets. Replace the top-8 / top-12 truncation (http.py:102, tls.py:78, dns.py:66) with the top 64 values plus '__other__', so the values sum to the true count. SNI and qname sets are additionally emitted at eTLD+1 (lib/names.py public-suffix subset: com.cn, gov.cn, edu.cn, org.cn, co.uk, ...).
3) New metrics:
   - l4flow: l4.dport_set {port: n}, l4.peer_set {peer /24 or service host: n}, each top 64 plus __other__; l4.syn_count; l4.pkts_total; l4.flow_duration_ms_avg.
   - http: http.status_3xx, http.get_count, http.write_count (POST/PUT/PATCH/DELETE), http.req_bytes_avg.
   - dns: dns.txt_count, dns.qname_len_avg.
   - tls: tls.handshake_ms_avg.
4) Freshness contract. Emit only for entities that had observations in this tick, always stamped ts = ctx.now (as http.py already does). store.add_raw updates first_seen and last_seen from that ts.
5) Pseudo-entity guard. Drop observations whose entity starts with '__' or 'class:', and count them in the health record. Otherwise add_raw would register them (store.py:49-53).

**Reads.** Observations, including extra.count, extra.bytes_*_total, extra.retransmits_total and extra.ts_sample

**Writes.** Existing l2l3/l4/http/tls/dns metrics with full sets; l4.dport_set, l4.peer_set, l4.syn_count, l4.pkts_total, l4.flow_duration_ms_avg, http.status_3xx, http.get_count, http.write_count, http.req_bytes_avg, dns.txt_count, dns.qname_len_avg, tls.handshake_ms_avg

**Perf.** Adds less than 1 ms per tick at 2k observations.

**Unit test.** (a) Feed 100 observations over 40 paths, one of them used once. Assert that path is in http.top_paths and that the set values plus __other__ sum to 100.
(b) One aggregated observation with count=50 must add 50 to http.requests and 50 to its path.
(c) An observation for entity '__system__' produces no RawMetric, and store.entities() is unchanged.
(d) l4.dport_set contains every port used.

## R2 — ActionTokenEngine [new]

- **File:** `backend/app/engines/raw/action_token.py`
- **Layer / order / interval:** raw / 6 / 1

**Purpose.** Learn a fine-grained action vocabulary per system with no host agent: templated (channel, op, resource, outcome) tokens, a time-ordered stream that carries stack ids, object ids, a stream of events to rare destinations, and zero-fill for silent entities.

**Algorithm.**

1) Tokens:
   - HTTP: METHOD host template(path) '|' status class.
   - TLS without L7: SNI eTLD+1 ':' dport plus size class u⌊log2 up⌋/d⌊log2 down⌋.
   - DNS: qtype plus templated qname. Labels left of the registrable domain become {rnd} if entropy ≥ 3.2, length ≥ 20 or digit ratio ≥ 0.3.
   - L4 only: proto/service(dport) plus size class.
   Templating uses Drain-lite (lib/template.py):
   - Masks: digits→{num}, UUID→{uuid}, hex of 16+ chars→{hex}, date→{date}, email→{email}, base64url of 16+ chars or char-entropy ≥ 3.5→{tok}, alphanumeric of 8+ chars with ≥ 3 digits→{id}. Query strings keep only their sorted parameter names.
   - A prefix tree per (system, host, method), depth ≤ 6. A node with more than 40 children whose top child has less than 0.5 share becomes {var}. Siblings with count ≤ 2 merge lazily.
   - The vocabulary is capped at 4000 per system; Space-Saving evicts to {rare}.
2) Stream rows are [ts, token_id, outcome, up, down, dest_id, stack_id]. stack_id comes from lib/stack.py, the same pure function R3 uses.
   - Keep at most 512 rows per entity per tick.
   - When a tick has more, keep whole sessions (split at intra-tick gaps > 30 s) chosen by reservoir sampling over session blocks, and write act.stream_frac = kept/total so consumers can reweight.
3) Rare destinations. R2 keeps its own decayed count of distinct entities per destination per system (half-life 7 d).
   - For destinations used by at most 20% of the system's entities, every event goes to act.rare_events {dest_id: [[ts, up, down]]}, up to 256 per destination per tick and outside the 512 cap.
4) Object ids. Values at {num}, {id} and {uuid} positions go to act.objs {template: {n: distinct count this tick, ids: up to 256, hll: 1024-byte HLL registers (p=10) when n > 256}}.
5) act.events = Σw for the entity.
   - Zero-fill writes act.events = 0 for each real entity (store.entities excludes pseudo-entities) that was seen within the last 30 d and has no observation this tick.
   - The zero-fill uses add_raw(touch=False), so last_seen is not refreshed.
6) Aggregated observations: tokens are counted with weight w. Timestamps come from obs.extra['ts_sample'] (at most 64 offsets), and act.stream_frac accounts for the rest.
7) The vocabulary is stored with put_model(system, '__system__', 'model.template').

**Reads.** Observations (per-event ts, extra.count, extra.ts_sample); store.entities, store.last_seen

**Writes.** act.tokens (up to 256 plus __other__), act.stream, act.stream_frac, act.events, act.objs, act.distinct_templates, act.new_template_ratio, act.rare_events; model.template@(s,__system__)

**Perf.** ≈2.6 µs per request (measured); at most 4 ms per tick

**Unit test.** (a) Feed 3000 /orders/view/<id> plus 2000 random 8-character 404 paths. Assert at most 60 templates, and that /orders/view/{num} holds at least 95% of its mass.
(b) A tick with 2000 events yields at most 512 rows, stream_frac equals kept/2000, and no session is split at the cut.
(c) A destination used by 1 of 10 entities appears in act.rare_events with all of its events.
(d) A silent known entity gets act.events=0 at ctx.now, and its last_seen is unchanged.
(e) act.objs lists the ids, with n = 2000 and HLL error within 5%.

**Integration notes (as built, docs/lib3/integration.md).**
- act.new_template_ratio is written only once model.template is ≥ 24 h old (m_template.NEW_REF_S; model['born'] is the vocabulary's first tick). The first tick of an empty vocabulary gives 1.0 for every entity, which B03 learnt into the buckets of that day-type; on the first live day of the mini pack every entity then scored new_template_ratio z ≈ −4 to −8 for hours.

## R3 — ClientStackEngine [new]

- **File:** `backend/app/engines/raw/client_stack.py`
- **Layer / order / interval:** raw / 7 / 1

**Purpose.** Produce passive client and device fingerprint tokens with timestamps, for identity, impersonation and linking.

**Algorithm.**

1) For each observation, build a stack token with lib/stack.py:
   ja3n | ua_family/major | os_family | ttl_class | tcpwin_class
   - ja3n is the JA3 string with its extension list sorted, so it survives Chrome's extension randomisation. Use JA4 when obs.extra['ja4'] is present.
   - UA family, major version and OS come from regexes.
   - ttl_class is the observed TTL rounded up to one of {32, 64, 128, 255}.
   - tcpwin_class comes from win_size buckets.
2) Emit client.stack_set {token: {n, bytes, first_ts, last_ts}} and client.stack_events [[stack_id, first_ts, last_ts, n]]. The timestamps come from obs.ts, so concurrency within 5 minutes can be tested at any cadence.
3) Also emit the component sets client.ua_set, client.ja3n_set and client.ttl_set, plus client.os_ua_ttl_pairs, which B09 uses for its consistency check.

**Reads.** Observations (ja3, user_agent, ttl, win_size, extra.ja4, ts, extra.count)

**Writes.** client.stack_set, client.stack_events, client.ua_set, client.ja3n_set, client.ttl_set, client.os_ua_ttl_pairs

**Perf.** Less than 1 ms per tick

**Unit test.** (a) Two JA3 strings that differ only in extension order map to one ja3n.
(b) A python-requests UA with TTL 57 yields a token ending '...|python-requests/2|linux|64|...'.
(c) Two stacks with interleaved obs.ts inside one 900-s tick yield stack_events whose [first_ts, last_ts] intervals overlap.

## D0 — AggregationEngine/PeriodicityEngine/TrendEngine [upgrade]

- **File:** `backend/app/engines/derived/{aggregation,periodicity,trend}.py (+ pure helper derived/fresh.py)`
- **Layer / order / interval:** derived / 0 / 1

**Purpose.** Compute window descriptors on a zero-filled wall-clock grid, so they describe true activity including idle time, and stop re-stamping stale inputs.

**Algorithm.**

1) Helper fresh.grid(store, s, e, name, now, span_s, kind) returns one value per tick in (now − span_s, now]. It reads store.raw_tail (islice) instead of raw_series, which copies the whole deque (store.py:80-86).
   - A tick with no raw point is 0 for counter kinds, confirmed by act.events == 0 at that tick.
   - Such a tick is NaN for gauge kinds.
2) aggregation: the span becomes 6 h wall-clock, replacing 30 points (aggregation.py:30,41). Statistics skip NaN. Nothing is emitted when the entity had no observation in the span.
3) periodicity: runs on the zero-filled 6-h count grid. This fixes periodicity.py:43, which reads only emitted points, so gaps collapse. The lag is reported in seconds. timing_regularity = 1 − CV over active ticks only.
4) trend: split-half test over a 12-h wall-clock grid.
5) Every output carries ts = ctx.now and dims {span_s, n_active}. FEATURE_SPEC v2 marks these features kind 'window', which is valid on idle ticks.

**Reads.** http.requests, l4.flows, dns.queries and the other configured raw metrics via raw_tail; act.events

**Writes.** derived.<metric>.{sum,mean,p95,max,cv}, derived.periodicity_score, derived.beacon_lag (in seconds), derived.timing_regularity, derived.<metric>.{ewma,slope,changepoint}

**Perf.** At most 3 ms per tick. The islice read replaces full deque copies, which the audits measured at 43 ms per tick for aggregation.

**Unit test.** Set http.requests on 1 of every 4 ticks for 96 ticks at Δt = 900, with act.events zero-filled.
- The aggregation mean must be about 0.25 × the active value.
- periodicity must report lag = 3600 s with score > 0.5.
- An entity silent for more than 6 h gets no window metrics.

**Integration notes (as built, docs/lib3/integration.md).**
- Activity gate: all three engines skip an entity whose last_seen is more than 6 h old (for trend this is narrower than its 12 h span, so an entity silent > 6 h gets no window metrics).
- Counters are rescaled to the current tick (v · dt_now / dt_tick): sum is the true total, mean the time-weighted rate per current tick; trend's EWMA uses a 1800 s half-life and reports its slope per current tick.
- Trend changepoint: Welch z of the recent-half mean minus the prior-half mean, split at now − 6 h, variance floored (Poisson for counters, (1 % of level)² for gauges), clipped to ±10.
- Periodicity: beacon_lag = 0 when the autocorrelation has no peak; act.events is a fourth count target; timing_regularity needs ≥ 2 active ticks.
- Retention: the D0 inputs are kept 24 h and the trend targets 13 h (contract B; store defaults).

## D1 — RatioEngine/EntropyEngine/GraphEngine [upgrade]

- **File:** `backend/app/engines/derived/{ratio,entropy,graph}.py`
- **Layer / order / interval:** derived / 1 / 1

**Purpose.** Emit instant derived metrics only when their inputs are fresh, so that absence reaches library 3 as absence.

**Algorithm.**

1) Freshness gate. fresh.fresh_raw(store, s, e, name, now) returns the value only when latest.ts == ctx.now. Today ratio.py:36-58, entropy.py:27-50 and graph.py:31-58 read latest_raw regardless of age and re-emit on every tick. After the fix, an instant metric is emitted only when all its inputs are fresh. Otherwise nothing is written.
2) ratio: a zero denominator means no emission, replacing the 0.0 at ratio.py:52. Each ratio also writes derived.<name>.n, its denominator, so exposure is available downstream.
3) entropy: also writes derived.<x>_n, the number of items. Entropies are computed on the full 64-entry sets.
4) graph:
   - _seen (graph.py:29) never decays and learns attacker peers forever. Replace it with a decayed dict {peer: last_seen_ts} that expires after 30 d.
   - new_peer_count counts peers not seen for 30 d.
   - peer_novelty and fanout are still emitted for lib-4 compatibility, but both are removed from FEATURE_SPEC.

**Reads.** http.*, l4.*, l3.*, dns.*, tls.* sets via fresh_raw

**Writes.** derived.http_error_rate/5xx/success/upload_dominance/bytes_per_flow/req_per_peer/dns_fail_rate with their .n; derived.sni_entropy/path_entropy/dns_name_entropy/dns_dga_score/ja3_diversity with their _n; derived.fanout, derived.peer_novelty, derived.new_peer_count, derived.dest_concentration

**Perf.** At most 2 ms per tick

**Unit test.** (a) Write raw metrics at t0 only and run the engine at t0 and at t0+900. derived.http_error_rate must exist at t0 and not at t0+900.
(b) A zero denominator writes nothing.
(c) graph: a peer seen at t0 and again at t0 + 31 d counts as new.

**Integration notes (as built, docs/lib3/integration.md).**
- derived.<entropy>_n is the total set mass including '__other__'; entropies are computed over the named entries only (contract B). Graph peers are eTLD+1 names plus l4.peer_set buckets.

## D2 — SessionEngine [upgrade]

- **File:** `backend/app/engines/derived/session.py`
- **Layer / order / interval:** derived / 4 / 1

**Purpose.** Fix the dead think_time and duty_cycle features using a wall-clock grid and real intra-session gaps, with correct freshness semantics.

**Algorithm.**

1) duty_cycle = fraction of active ticks over the last 24 h, taken from the act.events grid. Kind is window. This replaces session.py:40-44, which counts only emitted points.
2) Sessions are built from act.stream timestamps with gap threshold G, the entity's model.seq session_gap if present, else 30 min.
3) think_time_s_avg = median within-session inter-event gap observed in this tick. It is emitted only when the tick has at least 2 such gaps, so it is fresh and of kind avg. Otherwise it is not written. This replaces session.py:59-60, which uses tick-index gaps times window_s.
4) req_per_session and session_count are computed over sessions completed in the last 24 h (kind window). Counts are reweighted by act.stream_frac.

**Reads.** act.events, act.stream, act.stream_frac, model.seq (session_gap)

**Writes.** derived.session_count, derived.req_per_session, derived.think_time_s_avg, derived.activity_duty_cycle

**Perf.** Less than 1 ms per tick

**Unit test.** Make act.events active on 1 tick in 4 over 96 ticks, with 8-s gaps in act.stream.
- duty must be 0.25 ± 0.02.
- think_time must be ≈ 8 s on active ticks and absent on idle ticks.
- The window metrics must be written on every tick.

**Integration notes (as built, docs/lib3/integration.md).**
- req_per_session is absent (not 0 / NaN) while no session has completed in the 24 h window; duty is time-weighted across cadence changes; session counts are not reweighted by stream_frac (only event counts are).

## B01 — FeatureVectorEngine [upgrade]

- **File:** `backend/app/engines/behavior/feature_vector.py`
- **Layer / order / interval:** behavior / 1 / 1

**Purpose.** (Previously B1.) Build a per-tick representation that is presence-aware, carries exposure, is independent of cadence, covers numeric and categorical evidence, and includes time context. It never reads history.

**Algorithm.**

1) Activity: feature.active = 1 iff store.last_seen(s, e) == ctx.now, i.e. the entity had some observation in (now−Δt, now]. This replaces the Δt/2 rule, which fails when events fall in the first half of a tick.
2) Build each FEATURE_SPEC v2 entry by its kind (contract A), reading sources with latest_fresh.
   - count/bytes: stale → 0, otherwise log1p(v·60/Δt).
   - ratio: needs a fresh k and n > 0, else NaN.
   - avg: needs fresh and n ≥ 1, else NaN.
   - bounded: needs n ≥ 5, else NaN.
   - window: valid if written this tick.
   - clr: comp = CLR(x+0.5) over the counts [get, write, 4xx, 5xx, dns, tls, flows, syn].
   - intensity = log1p(act.events·60/Δt).
3) Exposure: feature.expo = {http: requests, dns: queries, tls: handshakes, flows: flows, probe: probes}.
4) Time context from timebins(now, tz, calendar): hour_local, dow, day_type (holiday and 调休 aware), bin48, bin168, the 15-minute slot, daypart ∈ {wd_day, wd_night, nwd_day, nwd_night}, and cadence class cc = the nearest of {60, 300, 900, 3600} to Δt.
5) Sketch (fixed formula). v[ns·16 + b] = Σ over tokens tok with h(tok) = b of s(tok)·sqrt(c_tok/C_ns), where C_ns = Σc in that namespace. Namespaces: act.tokens, client.stack_set, SNI eTLD+1, DNS eTLD+1, l4.dport_set. A non-empty namespace block has unit L2 norm; an empty one is all zeros.
6) Write feature.vec, nat, expo, active, tctx and sketch with store.add_vec. feature.<name> scalars are virtual views of feature.vec, not copies. Writing only vectors removes the 45 × 20000-object duplication.
7) Profile: fingerprint = vec (NaN → None); stable = model.baseline n_eff ≥ 96 and calibration healthy; updated = now. Remove the 12-sample flag at feature_vector.py:76.

**Reads.** Raw and derived metrics via latest_fresh; act.*; client.stack_set; store.last_seen; ctx.config tz/calendar/daypart_day_hours; model.baseline (n_eff only)

**Writes.** feature.vec, feature.nat, feature.expo, feature.active, feature.tctx, feature.sketch, virtual feature.<name>; profile.fingerprint/stable/updated

**Perf.** ≤0.3 ms per entity; ≤12 ms per tick at 40 entities

**Unit test.** (a) Staleness. With http.requests written at t0 only, running at t0+900 gives http_requests = 0, http_write_ratio NaN and active = 0.
(b) Sub-tick regression. If all events have obs.ts = tick_start + 5 s, then active = 1.
(c) Time context. With tz Asia/Shanghai, ts = 2026-10-01T02:00Z and that date in calendar.holidays: hour_local = 10, day_type = nonworkday, daypart = nwd_day.
(d) Cadence invariance. 15× the counts at Δt = 900 versus Δt = 60 give an equal vec.
(e) Sketch. Tokens with opposite hash signs give no NaN, and each non-empty namespace block has norm 1 ± 1e-9.
(f) derived_series('feature.bytes_up') returns the same values as column 0 of feature.vec.

**Canonical grain mode (spec v2.1, cadence.md §2–§4).**
- Every tick B01 also writes the 47 additive parts `feature.part` and the rolling rows `feature.live.<g>` (g ∈ {h, q} where observable). On a grain's decision tick it writes the scored rows `feature.nat.<g>`, `feature.vec.<g>`, `feature.meta.<g>`, `feature.expo.<g>` (and `feature.sketch.h`); their presence at ts = now is the decision flag.
- Set features are unions of the raw SetSketch series (`l4.peer_ids`, `l4.dport_ids`, `tls.ja3_ids`, `act.template_ids`), map features merge the raw count maps over the window, and span features are read on their 6-h / 24-h span decision ticks only. A set / map value needs coverage ≥ 0.95 G.
- The row's tctx is the window midpoint (`grains.row_tctx`). Coverage comes from B01's own tick log. The tick-level `feature.vec` / `feature.nat` are still written every tick (tick-native detectors, lib-4 compatibility).

## B02 — PeerGroupEngine [upgrade]

- **File:** `backend/app/engines/behavior/peer_group.py`
- **Layer / order / interval:** behavior / 2 / 1 (internal refit stride: 16 ticks or 6 h; the cold-start path runs every tick)

**Purpose.** (Previously B3; replaces clustering.py.) Discover a two-level class hierarchy (role class, then individual sub-class) with stable ids and soft membership. Also handle static CIDR classes and pool classes, cold-start typing, and class-transition and lineage events. The role is 'the class'.

**Algorithm.**

Runs when 16 ticks or 6 h have passed, whichever comes first. A light cold-start path runs every tick.
1) Static classes come from ctx.config.ip_classes, matched with the ipaddress module. Pools are CIDRs with at least 5 short-lived IPs, all of the same role, none of them linkable.
2) Eligible for dynamic clustering: at least 48 committed active ticks, not quarantined, no open incident.
3) Role descriptor per (system, ip). It is coarse and does not depend on the individual:
   - automation index A = mean(timing regularity, periodicity, non-browser UA share, 1 − think-time percentile, 1 − path entropy);
   - channel-mix CLR over http/dns/tls/flows;
   - template-family distribution, where family = channel|method class|host|first path segment, from model.vocab;
   - normalised 48-bin rhythm shape from model.rhythm;
   - volume tier;
   - device class: browser, library or other.
4) Role distance: D_role = 0.3|ΔA| + 0.2·Aitchison(channel)/norm + 0.2·√JSD(families) + 0.2·(1 − cos rhythm) + 0.1·[device class differs].
5) Level 1 (role): run HDBSCAN(metric='precomputed', min_cluster_size=2, min_samples=1, cluster_selection_method='eom', allow_single_cluster=True) on D_role, pooled org-wide.
   - Noise becomes a singleton 'unique' role plus a peer_outlier note.
   - If more than 30% of entities are noise, fall back to average-linkage agglomerative clustering cut at the largest merge-height gap.
   - Super level: human versus machine by A, with hysteresis 0.4 / 0.6.
6) Level 2 (sub-class) inside each role:
   - D_ind = 0.4·mean_f W1(deciles of the entity's predictive) + 0.3·√JSD(full templates, SNI, dports) + 0.2·(1 − cos rhythm168) + 0.1·[stack differs];
   - HDBSCAN with cluster_selection_method='leaf';
   - class_path = super/role/sub.
   This is why individual Dirichlet path preferences split sub-classes but not roles.
7) Stable ids: Hungarian matching on 1 − Jaccard(members). Inherit the id when J ≥ 0.3, otherwise mint a new one and emit class_split or class_merge. A class retires after 3 absent runs. Role ids are global. Class state is keyed per system as (s, 'class:<rid>').
8) Names come from data in natural units: human or automated, dominant action family, active window, volume tier, and the top-3 class-vs-rest Cohen's d features.
9) Soft membership: P(c|e) ∝ exp(−D_role(e, medoid_c)/d90_c).
10) Cold start, every tick, for entities with fewer than 48 commits and at least 3 active ticks:
   - build a provisional descriptor from the raw vocab and rhythm seen so far;
   - if D_role ≤ d90_c for the nearest role, emit new_entity_matched (INFO); otherwise emit new_entity_unmatched (MEDIUM).
11) class_transition fires when the argmax role changes with p ≥ 0.7 on 3 consecutive runs.

**Reads.** model.baseline, model.vocab, model.rhythm, model.client, model.timing, ctx.config.ip_classes, behavior.quarantine, store.incidents(status=open)

**Writes.** model.class@(__org__,__org__); class profiles (s, class:<rid>|class:static:*|class:pool:*); profile.archetype (= class_path), profile.archetype_confidence, profile.extra.peer_group; events new_entity_matched/unmatched, class_transition, class_split/merge, peer_outlier

**Perf.** ≈1 ms per HDBSCAN level (measured) plus ≈8 ms of distances per run; ≤1 ms per tick amortised

**Unit test.** Build 3 roles × 4 individuals. Each individual has its own Dirichlet(α = 0.4) template preferences and private paths, but they share families and rhythm shape within a role. Add 1 outlier.
- Level-1 ARI against roles must be 1.
- Fewer than 10% of the entities may be singletons.
- Level 2 must give at least 3 sub-classes in some role.
- The outlier must be 'unique'.
- Perturbing one member's statistics by 5% leaves the ids unchanged.
- A new entity with a role-2 profile and 3 active ticks produces new_entity_matched with prob ≥ 0.8.

**Integration notes (as built, docs/lib3/integration.md).**
- Engine interval 1 with an internal refit stride (16 ticks or 6 h); the cold path runs every tick. HDBSCAN epsilon is applied as a post-merge (0.15 roles, 0.05 subs); d90 is floored at 0.15.
- assign carries the per-IP automation index A (B30 prefers it); linked entities are read through lib/m_link (active links only).

## B03 — BaselineEngine [upgrade]

- **File:** `backend/app/engines/behavior/baseline.py`
- **Layer / order / interval:** behavior / 3 / 1

**Purpose.** (Previously B2.) Per-entity seasonal, distributional baseline that is exposure-aware, with two anchors (a rate-capped current anchor and a slow reference anchor) and hierarchical priors that end in organisation-level hyperpriors. It learns only through trust-gated, delayed, checkpointed and reversible commits.

**Algorithm.**

0) Hyperpriors (lib/priors.py) give a finite, wide predictive from the first tick, so cold start cannot deadlock.
   - count: Gamma-Poisson with μ0 = 1 event/min and shape a0 = 0.5.
   - ratio: Beta(0.5, 0.5).
   - bytes/avg/bounded/clr/window/gauge: NIG with a kind default m0 (bytes: ln 1e4), κ0 = 0.01, α0 = 1 and β0 = 4.
1) Buckets are bin48, switching to bin168 after at least 4 weeks of commits. Each commit is spread over neighbouring hours with von Mises weights exp(4(cos(2π·dh/24) − 1)), |dh| ≤ 2.
2) Sufficient statistics per kind, decayed by γ = 2^(−Δt/H):
   - count: [W, Σx, Σe, Σx², Σxe, Σe²], with e = exposure in minutes;
   - ratio: [W, Σk, Σn, Σk²/n, Σ(k/n)²];
   - other kinds: NIG on the transformed value, weighted by min(1, n/5).
   Every 96 commits, H is chosen from {7, 14, 28} d by the pinball loss of p5/p95.
3) Hierarchy: entity → role class (s, class:<rid>) when the system has at least 3 members, otherwise system → org → hyperprior. The backoff pseudo-counts κ are set by empirical Bayes, κ = m(1−m)/var_between − 1, clipped to [2, 50]. The class, system and org tiers update from the same gated rows.
4) Two anchors:
   - Current: every bucket mean may move at most 0.1σ15 per day, where σ15 is the predictive sd for a 15-minute exposure. The cap does not apply when rebased.
   - Reference: commits with a 24-hour delay, only rows with trust == 1 and with no incident or regime within ±24 h, capped at 0.03σ15 per day plus model.control.allow_drift. It becomes the golden anchor (median of up to 4 weekly snapshots from weeks with max risk < 30) once one exists.
5) Commits (lib/gating.py): commit row t−D with weight trust(t−D), where D = max(4 ticks, 600 s); if quarantined, hold it. Record w_eff and w_prov.
6) Checkpoints: current anchor hourly and reference anchor daily, float16, geometric retention.
7) Honour model.control: rebase_from (replay held rows with w_prov into version+1, which resets the reference after 1 d of the new regime), rollback_to, release, frozen and allow_drift (contract H).
8) Publish put_model('model.baseline') for the entity, class, system and org. Fill profile.baseline_median, baseline_mad and seasonal at the current bucket for compatibility, and set profile.extra.maturity.

**Reads.** feature.nat, feature.expo, feature.active, feature.tctx; behavior.trust, behavior.trust_prov, behavior.quarantine; behavior.risk; model.class; model.control; store.incidents

**Writes.** model.baseline@(s,e|class:<rid>|__system__), model.baseline@(__org__,__org__), checkpoints 'baseline.current' and 'baseline.reference', profile.baseline_median/mad/seasonal, profile.extra.maturity

**Perf.** ≤0.15 ms per entity per tick (numpy over 5 buckets × 52); checkpointing ≤0.5 ms per tick amortised; ≤7 ms per tick at 40 entities

**Unit test.** (a) Feed 200 ticks of Poisson(10/min) with trust 1, then 40 ticks of Poisson(300) with trust 0. The current mean must stay within 5% of 10.
(b) A 10× step with trust 1 moves the current anchor at most 0.1σ15 per day and the reference at most 0.03σ15 per day.
(c) Deadlock regression: an empty store gives the very first entity a predictive with p(10 req/min) > 0.05.
(d) Commits lag ctx.now by exactly D.
(e) Rollback. Commit 100 clean rows, then 20 attack rows with trust 1, then set control.rollback_to = t100. The statistics must equal a fit on rows ≤ t100 within 1e-6, and the 20 rows must be held.
(f) A new entity with no commits returns the class predictive mean.

**Integration notes (as built, docs/lib3/integration.md).**
- Accessor module lib/m_baseline (helpers_api index): float32 checkpoints; a band-day cap on the data mean; location-only bin168; retransmit_rate and probe_loss as t family; per-family stats layout (k = 6 / 4 / 3); reference = golden once one exists.
- own_support(model, tctx) exposes the entity's own evidence at a bucket (B14 gates its charts on it); n_eff_by_bucket serves B30.

**Canonical grain mode (spec v2.1, cadence.md §6).**
- Learners: `baseline.current` and `baseline.reference` on H decision rows, `baseline.current.q` on Q decision rows; a row's commit weight is the minimum trust over its window; exposure is the row's coverage.
- The Q predictive is the Q native current anchor plus κ_T = 16 conjugate pseudo-rows of the H predictive transferred with v_f = 1 + ((m−1)/m)·ω_f (NB r/v, BB c through v, t scale √v and a location δ_f). ω_f and δ_f are learnt on paired hours and EB-shrunk entity → class → system → org (default ω = m). The Q reference is the transferred H reference (golden once it exists). A Q score is provisional while the native share π_nat < 0.5.
- σ15 of the rate caps comes from the Q transfer of the bucket's H posterior, so the caps keep their architecture meaning. Accessors: `m_baseline.predictive_set(..., grain=g)`, `transfer_pred`, `pseudo_stats`, `omega_chain`.

## B04 — LikelihoodEngine [new]

- **File:** `backend/app/engines/behavior/likelihood.py`
- **Layer / order / interval:** behavior / 4 / 1

**Purpose.** (Previously B5; replaces the robust-z primary in anomaly.py.) Exact, exposure-aware predictive p-values for every feature against both anchors and against the class, together with the normal scores used by the multivariate and sequential engines.

**Algorithm.**

Runs only when feature.active == 1. Absence is handled by B07. For each feature and its current bucket (bayes.py):
- count: NB with mean μ̂·e and size r = 1/(1/κ̂ + 1/a). κ̂ is a moment estimate of overdispersion clipped to [0.5, 1e3]; a is the posterior shape. CDF and SF use scipy.special.betainc, never scipy.stats.nbinom (286 µs versus 3 µs measured).
- ratio: BetaBinomial(n*, p̂·c, (1−p̂)·c) with c = min(Σn + 2, φ̂). φ̂ is the method-of-moments concentration from committed rows, clipped to [20, 1000]. Exact gammaln pmf for n* ≤ 200, normal approximation above. If n* = 0 the feature is NaN, not p = 1.
- avg/bytes/bounded/window/gauge: Student-t from the NIG.
Per feature:
- the two-sided mid-p is computed against the current anchor (p_cur) and against the reference anchor (p_ref), and p_f = min(1, 2·min(p_cur, p_ref));
- z = Φ⁻¹(u_cur) and zr = Φ⁻¹(u_ref).
Channels:
- wHMP within each dependence group from model.groups (weight 1/|group|), then wHMP across groups;
- intensity channel = volume group → score.marg_int = −log10 p;
- the other groups → score.marg_shape;
- the same calculation under the class-bucket predictive → score.peer.
Axes: the groups whose p < 0.01 are written to behavior.axes.
A new entity uses the backoff predictive, so it is scored from its first tick. The model p-values go to behavior.pm for calibration's small-sample prior.
profile.extra.model_state = p5/p50/p95 in natural units per feature for the current bucket, from the NB/BB/t ppf.

**Reads.** feature.nat, feature.expo, feature.active, feature.tctx; model.baseline (entity current and reference, class, system); model.groups

**Writes.** behavior.z, behavior.zr, behavior.pf, behavior.score[marg_int, marg_shape, peer], behavior.pm[...], behavior.axes, profile.extra.model_state

**Perf.** ≈0.15 ms per entity with two anchors (measured: NB 0.022 ms for 16 features, BB 0.031 ms per feature); ≤6 ms per tick

**Unit test.** Tail values below were checked with scipy.
(a) Poisson baseline of 22 per tick with κ̂ capped at 1e3:
   - observing 1 gives p_f < 1e-7 (P(X≤1) = 6.4e-9);
   - observing 3 gives p_f < 1e-5 (P(X≤3) = 5.7e-7) and z < −4.5.
   With NB r = 20, observing 3 gives p < 1e-3 (P = 1.04e-4).
(b) Ratio k = 1 of n = 2 against mean 0.1 with c = 50: p > 0.05 (sf = 0.188).
(c) k = 100 of n = 200 against mean 0.1: with φ̂ = 50, p < 1e-6 (sf = 1.3e-8); with φ̂ = 20, p < 1e-3 (sf = 7.7e-5).
(d) Over 2000 null draws, z has mean ≈ 0 and sd ≈ 1 (±0.1), and a KS test on p_f is not rejected at 0.01.
(e) Dual anchor: shift the current anchor by +1σ, leave the reference unchanged, and observe the reference + 3σ. p_ref drives p_f < 0.01 although p_cur ≈ 0.05.
(f) n* = 0 gives NaN.

**Canonical grain mode (spec v2.1, cadence.md §6.2, §7.1).** B04 scores H rows at H decision ticks (`marg_int`, `marg_shape`, `peer`, `behavior.z` / `zr` / `pf`) and Q rows at Q decision ticks (`marg_int_q`, `marg_shape_q`, `behavior.*.q`; no `peer_q`). `none`-transfer features (distinct counts, bounded map features) are scored at Q only once their native weight reaches κ_T, otherwise NaN; provisional Q scores write `behavior.prov`. `profile.extra.model_state` holds `{grains: {h, q}}` bands per hour and per 15 min.

**Perf pass (integration.md §9, results bit-identical).** All rows of a tick are scored in one batch (`m_baseline.predictive_set_many` / `predictive_q_many` / `quantiles_many`, the array NB mid-p kernel, grouped `bayes` BB quantiles); B04 is about 2.3× faster. The Beta-Binomial mid-p and the observation transform stay on their scalar paths: the vectorised versions differed by ~1e-11, enough to flip float32 ring values.

## B05 — CommonModeEngine [new]

- **File:** `backend/app/engines/behavior/common_mode.py`
- **Layer / order / interval:** behavior / 5 / 1

**Purpose.** (Previously B6.) Remove only benign-eligible common-mode movement from each entity's residual: org, system and class shifts in volume, transport and app-error groups (month-end, outage, WAN degradation). It is leave-one-out and never masks categorical, identity, c2 or exfil evidence.

**Algorithm.**

1) Eligible groups: volume, transport, app-error (5xx, latency) and probe. Never breadth, categorical, dns, tls, identity or composition.
2) For entity e, class c and eligible group g:
   L_{t,g}^(−e) = median over OTHER active, non-quarantined members of mean_{f∈g} z_{m,t,f}.
   This needs at least 3 other members. Otherwise fall back to the system level (at least 3 others), otherwise L = 0.
3) Loading β_{e,g} = (Σ L·z + 4)/(Σ L² + 4), a ridge shrink toward 1, refit every 16 ticks on committed rows.
4) zi = z − β·L for eligible groups; zi = z elsewhere.
5) Exclusions: flag = 0 and zi = z when the entity has any of the following this tick:
   - system-tier novelty;
   - a lib-4 match ≥ medium;
   - client or identity evidence with p < 0.01;
   - budget_exfil evidence.
   This stops a coordinated campaign from laundering itself.
6) behavior.common.flag[g] = 1 when |z_g| > 2 and |zi_g| < 1.
7) Coherence: when at least 50% of the system shows |z_g| > 2 in the same direction for 2 ticks or more, emit one system_shift (INFO). Also write behavior.common.<g> at '__system__' and 'class:<id>' for B18.

**Reads.** behavior.z, feature.active, behavior.quarantine, model.class, behavior.score (novelty, client, identity, budget_exfil at t−1), store.matches(since=t−Δt)

**Writes.** behavior.zi, behavior.common.flag, behavior.common.<group>@(__system__|class:<id>); event system_shift

**Perf.** ≤1 ms per tick

**Unit test.** (a) 6 class members all shift +3 in the volume group, and one member has a new system-tier SNI. The other 5 get |zi_volume| < 0.5 and flag 1. The member with the new SNI keeps zi = z with flag 0.
(b) In a 4-member class where only the scored entity shifts +3, its zi_volume stays ≥ 2.8 (leave-one-out).
(c) A shift in the app composition group is never removed.
(d) Exactly one system_shift is emitted.

**Canonical grain mode.** The H pass runs at H decision ticks on H z rows and the Q pass at Q decision ticks (learner `common_mode.q`); `behavior.common.<g>` and `behavior.zi` are written per grain. A group with no scored member (or a class with no active member) is written as undefined (L NaN, n 0, dir 0) every tick (integration.md §4).

## B06 — MultivariateEngine [upgrade]

- **File:** `backend/app/engines/behavior/multivariate.py`
- **Layer / order / interval:** behavior / 6 / 1

**Purpose.** (Previously B7; replaces the IsolationForest that anomaly.py:117-139 refits on every run.) Correlation-aware scoring of magnitude (Hotelling T²) and of correlation breaks (SPE), with exact per-feature contributions. It fits only on gated history and handles missing dimensions correctly.

**Algorithm.**

Refit: every 16 ticks or 4 h, with a random phase per entity, on committed zi rows (trust-weighted, at most 336).
- Winsorise at ±4.
- OAS, then one C-step: keep the 75% of rows with the smallest d², rescale by median(d²)/χ²_{p,0.5}, reweight at χ²_{p,0.975}, and apply OAS again.
- eigh with an eigenvalue floor of 1e-3·mean. Keep the smallest k that explains 90% of the variance.
- Young entity: Σ̃ = (nΣ_e + 30Σ_class)/(n + 30), using model.density@class.
Every 32 ticks or 8 h: recompute the dependence groups (|Spearman ρ| > 0.8, average linkage) into model.groups.
Scoring on the observed dimensions o:
- T² = z_oᵀ Σ_oo⁻¹ z_o, using the Cholesky of Σ_oo cached per missingness pattern (LRU of 8, ≈20 µs per new pattern). p from the Hotelling prediction distribution: T² · n(n−q)/(q(n−1)(n+1)) ~ F(q, n−q), with q = |o|.
- SPE is computed on the completed vector. Missing dimensions are imputed by conditional expectation, z_m = Σ_mo Σ_oo⁻¹ z_o (Nelson, Taylor & MacGregor 1996). SPE = ‖z − U_kU_kᵀz‖², with p from the Box approximation g·χ²_h fitted on the training SPE.
- Reconstruction-based contributions RBC_f = (e_fᵀΣ⁻¹z)²/(e_fᵀΣ⁻¹e_f), each χ²₁. Axes are the groups of the top RBC features with p < 0.01.
- behavior.wh = the Wilson–Hilferty normal score of T².

**Reads.** behavior.zi, behavior.trust, model.class, model.density@class, model.control

**Writes.** model.density@(s,e|class:<rid>), model.groups@(s,__system__), behavior.score[t2, spe], behavior.axes, behavior.wh, profile.extra.mv_model, profile.extra.last_contrib

**Perf.** Refit OAS ≈1.9 ms at 600×64 (measured) per 16 ticks; score ≈10–30 µs; ≤5 ms per tick

**Unit test.** (a) Train with features 1 and 2 correlated at ρ = 0.95. Scoring x = (+2, −2) gives SPE p < 1e-4 with each |z| ≤ 2, and RBC ranks features 1 and 2 first. Scoring (+2, +2) gives T² p < 0.05 and SPE not significant.
(b) With feature 2 missing, x = (+2, NaN) gives a finite T² with q = |o|, and the imputed z2 ≈ 0.95·2.
(c) Under the null, over 2000 draws with n = 200, the T² p-values give KS D < 0.05; this exercises the prediction scaling.

**Canonical grain mode (cadence.md §6.4).** H: T² / SPE on H zi at H ticks, density refit every 16 H rows or 4 h on ≤ 336 committed H rows. Q: `t2_q` / `spe_q` on Q zi from `model.density.q`, shrunk to the H density Σ̃_Q = (nΣ_Q + 30Σ_H)/(n + 30) and provisional while n_Q < 64. `model.groups` is fitted from H zi only.

**Open (cadence.md §17 item 2).** The density commits zi rows computed while B04 still predicted from the hyperprior (the first D_min of a young entity), so on warm-ups shorter than D_min + 14 d (mini, smoke) T² is large after the switch to own predictives. Candidate fix: admit a zi row only when B04 scored it against an own-support predictive. Not changed.

## B07 — RhythmEngine [new]

- **File:** `backend/app/engines/behavior/rhythm.py`
- **Layer / order / interval:** behavior / 7 / 1

**Purpose.** (Previously B8.) Model when each IP and each class is active, on a fixed 15-minute slot clock that does not depend on cadence. Detect off-hours activity at normal intensity, unexpected silence of scheduled machines, and schedule shifts. Publish circadian descriptors.

**Algorithm.**

1) Slot clock. Each 15-minute local slot is active if any act.stream timestamp or active tick falls in it. A 3600-s tick is resolved into its 4 slots from the timestamps. At 60-s ticks, the current slot is updated provisionally when the first activity appears, since activity within a slot is monotone.
2) Model: P(active | bin) ~ Beta(a_b, b_b) per bin48, or bin168 after at least 4 weeks.
   - Updated at slot completion with delay D and weight equal to the slot's trust (minimum over its ticks).
   - von Mises smoothing across hours with κ = 4.
   - Class prior Beta(6·π_class,b, 6·(1−π_class,b)).
   - Forgetting half-life 28 d.
3) Off-hours: Bernoulli CUSUM in bits, only over bins with p̂ ≤ 0.3.
   - p1 = min(0.95, max(0.5, 5p̂)).
   - W = max(0, W + a·log2(p1/p̂) + (1−a)·log2((1−p1)/(1−p̂))), once per slot.
   - Bins with p̂ > 0.3 contribute 0, which avoids the invalid p1 > 1 case.
   - Threshold h = log2(ARL_slots) + 0.5 (Wald bound), with ARL = 100 days of slots: h ≈ 13.7 bits.
   - At p̂ = 0.02, each active slot adds +4.64 bits, so the 3rd active slot alarms.
   - p_eq = 2^(−W).
4) Silence:
   - Eligible only for machine-like entities and bins with p̂_b ≥ 0.95. Machine-like (W7 tuning, replaces "normalised 168-bin entropy ≤ 0.8", which held for most office workers and for no 24/7 client): the automation index m_rhythm.automation_index = mean of the identified components
     - off-hours ratio min(1, r_off / r_biz) (active rate outside workday 08–19 over the rate inside),
     - week ratio min(r_wd, r_nwd) / max(r_wd, r_nwd),
     - presence regularity 1 − Σ r·4r(1−r) / Σ r over the observed cells,
     - B02's per-IP automation index A (timing regularity, periodicity, non-browser share, think time, path entropy),
     with B02's hysteresis (enter ≥ 0.6, leave < 0.4), the rhythm components only from a model spanning ≥ 7 d, and never when regularity < 0.5. Validated on the generator population (integration.md §8.2): humans ≤ 0.34, machines ≥ 0.73.
   - s accumulates −ln(1 − p̂_b) over silent slots; p_eq = e^(−s); the alarm threshold comes from ARL = 100 days.
5) Schedule shift. When silence in the usual window is followed, within 24 h, by activity of the same duration and volume in a new window (circular shift > 1 h):
   - emit schedule_shift (LOW);
   - set behavior.axes = ['temporal'] with the 'shift_explained' flag, which caps the severity at LOW;
   - any change in destination or content is not capped.
6) Calendar self-healing: if system-wide presence in a bin exceeds 3× its usual level, treat that day as a workday.
7) Descriptors: μ_h, R, the 80% window, the weekday/weekend ratio. A class rhythm on the class active fraction is also published for B18.

**Reads.** feature.active, feature.tctx, act.stream (ts), behavior.trust, model.class, model.control

**Writes.** model.rhythm@(s,e|class:<rid>), behavior.score[offhours, silence], behavior.acc_alarm[offhours, silence], behavior.rhythm, behavior.axes, profile.extra.rhythm; event schedule_shift

**Perf.** ≤1 ms per tick

**Unit test.** (a) An entity active on workdays 09–18 for 20 days gets activity at 02:00. Three active slots give W = 13.9 ≥ h and an alarm, while a normal day gives W = 0. The same scenario at Δt = 60 and at Δt = 900 alarms in the same wall-clock slot with the same W.
(b) A backup active daily 02:00–02:40 that skips one night gives silence p_eq ≤ 1e-4.
(c) The same skip for a human-rhythm entity is ineligible.
(d) A bin with p̂ = 0.3 gives no NaN; a bin with p̂ = 0.5 contributes 0.
(e) A backup that moves from 01:00 to 03:00 with the same volume gives schedule_shift, and its temporal evidence is capped at LOW.

## B08 — NoveltyEngine [new]

- **File:** `backend/app/engines/behavior/novelty.py`
- **Layer / order / interval:** behavior / 8 / 1

**Purpose.** (Previously B9.) First-seen, rarity and distribution-shift analytics over everything each IP touches, at entity, class and system tiers. It stays quiet for exploratory humans and for benign rollouts, and it hands risky adoptions to the class monitor instead of dropping them.

**Algorithm.**

Dimensions: method+template from act.tokens, content type, SNI eTLD+1, DNS domain and qtype, dport, peer /24, and the lib-4 category.
State:
- decayed counts (entity 30 d), first_seen and last_seen per value;
- Space-Saving caps: K = 256 per entity, 2048 per class and per system;
- decayed document frequency df over 30 d;
- keyed per system; class state is under (s, class:<rid>).
Hierarchical Dirichlet backoff:
- p_e(v) = (c_e + 5p_c)/(N_e + 5);
- p_c(v) = (c_c + 5p_s)/(N_c + 5);
- p_s(v) = (c_s + U)/(N_s + 1);
- surprisal I_v = −log2 p_e(v).
Scores:
1) Tier flags FS_entity, FS_class and FS_system, plus re-emergence after 30 d.
2) Good–Turing novelty-rate test: p_new = (N1 + 0.5)/(N + 1); for k first-seen values among n accesses, p = betainc(k, n−k+1, p_new).
3) Class rarity IDF = ln((N_class+1)/(df+1)) + 1, multiplied by a sensitivity weight w ∈ [1, 3] from sensitive_patterns and from low-prevalence, heavy-byte values.
   score.novelty = max(−log10 p_rate, ω·Σw·I/12), where ω = 1 − unseen mass. Instantaneous.
4) score.novelty_rate: a daily negative-binomial test on the count of class-rare novelties (slow enumeration or scanning). Accumulator.
5) score.jsd: JSD of the decayed 1-h and 24-h windows against the smoothed profile, conformal. Accumulator.
Axes: categorical by default; exfil when the value is external (not in org_domains) and upload-dominant; privilege when it is sensitive or admin.
Adoption: if at least max(3, 30%) of the class first saw v within 7 d, the per-entity score is multiplied by 0.1 and class_adopted (INFO) is emitted. The adoption record, with flags {external, upload, sensitive, new_eTLD1_org}, is always written to model.vocab@class for B18. The per-entity discount therefore cannot hide a risky class-wide adoption.
Events: first_seen when the tier ≥ class or bits ≥ 12, and rare_access for sensitive values, deduplicated by (entity, dim, value).
Feedback allowlists apply. Values inherited through a link (model.link) are not new at the entity tier.

**Reads.** act.tokens, tls.sni_set, dns.qname_set/qtype_set, l4.dport_set/peer_set, act.stream (direction), store.matches(since=t−Δt), model.class, model.feedback, model.link, behavior.trust, model.control

**Writes.** model.vocab@(s,e|class:<rid>|__system__), behavior.score[novelty, novelty_rate, jsd], behavior.acc_alarm[novelty_rate, jsd], behavior.axes, profile.extra.categorical; events first_seen, rare_access, class_adopted

**Perf.** O(values); ≤3 ms per tick

**Unit test.** (a) A stable integration client with 1 template for 200 ticks sees one new template: p < 1e-3.
(b) An explorer with p_new = 0.3 sees 1 new template: p > 0.05.
(c) /hr/salary/export with df = 1 of 10: first_seen at class tier with bits ≥ 12, axis privilege.
(d) 6 of 8 members adopt /v2/orders (internal GET) in one day: class_adopted, and each entity's score is ≤ 10% of the undiscounted score.
(e) 4 of 6 members adopt an external upload destination: the per-entity score is discounted, but the adoption record has external = upload = True.

## B09 — ClientIdentityEngine [new]

- **File:** `backend/app/engines/behavior/client_identity.py`
- **Layer / order / interval:** behavior / 9 / 1

**Purpose.** (Previously B10.) Detect a different client stack on a known IP (impersonation, tool abuse, NAT) and tell it apart from browser upgrades and rollouts, using timestamped stack events.

**Algorithm.**

Model: hierarchical Dirichlet over stack tokens, decayed with a 14-d half-life.
- p(s|e) = (n_e + 5p(s|class))/(N_e + 5), backing off to the system.
- Stack surprise S = −log2 p, weighted by the token's share of traffic over the last 10 ticks.
Discriminators:
- Concurrency C: the dominant stack s0 (p > 0.5) and a new stack s1 have client.stack_events intervals that overlap or are within 5 minutes of each other, whatever the tick length.
- Replacement: s0 is absent for longer than its P99 gap.
- Inconsistency I: the UA-declared OS does not match the TTL class, or p(ja3n | UA family) < 0.01 in the system co-occurrence table.
- Rollout R: the share of the role class (and of the system) that acquired s1 within 7 d.
Risk = σ(0.6S + 3C + 2I − 4R − 3); score.client = −log10(1 − risk + 1e-6). Instantaneous, axis identity.
Events:
- client_impersonation (HIGH) if C or I holds and risk > 0.8;
- client_change (INFO) on a replacement when R ≥ 0.3 or the UA family is the same with a version increase.

**Reads.** client.stack_set, client.stack_events, client.ua_set, client.ja3n_set, client.ttl_set, client.os_ua_ttl_pairs, model.class, behavior.trust, model.control

**Writes.** model.client@(s,e|__system__), behavior.score[client], behavior.axes, profile.extra.client_stacks; events client_impersonation, client_change

**Perf.** ≤1 ms per tick

**Unit test.** (a) An entity on chrome126|win|128 for 100 ticks gains python-requests|linux|64, with events interleaved inside one 900-s tick: client_impersonation within 2 ticks.
(b) 80% of the class moves from chrome126 to chrome127 (new ja3n) in one day: only client_change INFO, and the client p > 0.01.
(c) A UA copied from chrome126 with a Linux JA3 and TTL 64: I = 1 and client_impersonation.

**Integration notes (as built, docs/lib3/integration.md).**
- Concurrency C requires the two stacks to interleave; replacement is measured on the entity's active clock; p(ja3n | UA) is keyed 'family/major'; the per-token surprise is capped at 20 bits; counts are slot-equivalents (share · dt/900 · trust).

## B10 — SequenceEngine [upgrade]

- **File:** `backend/app/engines/behavior/sequence.py`
- **Layer / order / interval:** behavior / 10 / 1

**Purpose.** (Previously B11.) Learn each IP's own grammar over fine-grained action tokens, generalised to its role class. Score sessions and windows relative to the entity's own entropy rate. Fixes the false positives, novel-state blindness and churning model keys at sequence.py:60 and 88-90.

**Algorithm.**

Streams:
- (A) act.stream tokens as template|outcome, reweighted by act.stream_frac. Tokens with a system count below 3 map to {rare:<channel>}.
- (B) the multi-label bag of lib-4 categories per tick (confidence ≥ 0.6, one-tick lag).
Sessions: the idle threshold G_e is the valley of a decayed 32-bin histogram of log gaps, clipped to [2 min, 2 h], and published as session_gap.
Model:
- PPM-C of order ≤ 3 per entity, with 30-d decay.
- On escape, back off to the role-class PPM keyed by (s, class:<rid>) — a stable id, never a name — then to the system unigram with Good–Turing novel mass N1/N.
- The real vocabulary size is used, and unseen source states are always scored.
Scores:
- windowed excess E = mean surprisal over W = 8 tokens − Ĥ_e (the entity's entropy rate);
- session NLL* = (L·NLL + 5μ_e)/(L + 5);
- within-session CUSUM on s_i − (μ_e + 0.5σ_e), h = 8 bits;
- semi-Markov dwell: NB per token family on run lengths, score.dwell = −log10 P(L ≥ ℓ);
- class_llr = log P_e − log P_class.
score.seq = max(E statistic, CUSUM/h·8). Instantaneous, evaluated at window and session boundaries.
Axes: sequence; credential when the token family is auth.
Learning uses gated commits and honours model.control.
Published: model.seq plus the pure helper ppm.loglik(model, tokens), which B16 uses to score candidates.

**Reads.** act.stream, act.stream_frac, store.matches(since=t−Δt), model.class, behavior.trust, model.control

**Writes.** model.seq@(s,e|class:<rid>|__system__), behavior.score[seq, dwell], behavior.axes, behavior.seq.class_llr, profile.extra.sequence

**Perf.** ≈3–8 µs per token; ≤5 ms per tick

**Unit test.** (a) Train on 300 sessions of login→dashboard→orders→view{n}. The session login→export→login→export gives excess ≥ 4 bits; a normal held-out session gives < 1 bit.
(b) A high-entropy random-walk entity on normal traffic has a false-positive rate ≤ 1% at the conformal 0.01 threshold.
(c) A never-seen source token gets surprisal > 0.
(d) Renaming the role name keeps the same model (keyed by id).
(e) With stream_frac = 0.25, the counts are weighted ×4.

## B11 — TimingEngine [new]

- **File:** `backend/app/engines/behavior/timing.py`
- **Layer / order / interval:** behavior / 11 / 1

**Purpose.** (Previously B12.) A fine-grained timing profile per IP: the inter-arrival distribution, burstiness, memory and think time. It separates humans from scripts and feeds identification.

**Algorithm.**

From act.stream timestamps, using only intra-tick sampled blocks so that gaps are not distorted by sampling:
- A decayed 32-bin histogram of log2 gaps from 10 ms to 1 day (τ = 7 d). score.timing = JSD of the recent 6-h histogram against it, conformal. Accumulator, axis temporal.
- Burstiness B = (σ − μ)/(σ + μ) and memory M = corr(Δ_i, Δ_{i+1}), via exponentially weighted Welford updates.
- Think time: a log-normal fit to within-session gaps.
- A binned rfft over 5-s bins of the last 2 h, whose peak is confirmed by a Rayleigh test. This is only for strictly periodic trains; renewal-jittered beacons are left to B12.
Descriptors are written to behavior.timing for B15, B16 and B30.

**Reads.** act.stream, act.stream_frac, model.seq (session_gap), behavior.trust, model.control

**Writes.** model.timing@(s,e), behavior.score[timing], behavior.acc_alarm[timing], behavior.timing, profile.extra.timing

**Perf.** ≤2 ms per tick

**Unit test.** (a) Log-normal gaps with σ = 1 give B ∈ [0.2, 0.6].
(b) A constant 0.8-s scraper gives B < −0.8 and timing p < 1e-3 against the human model.
(c) A 37 s ± 0.3 s train over 2 h gives a period in 35–40 s with Rayleigh p < 1e-4.

**Integration notes (as built, docs/lib3/integration.md).**
- Unit test (a) reads: human sessions with LogNormal(ln 8 s, σ = 1) think time and minutes-long breaks give B in [0.2, 0.6] (a pure log-normal renewal with σ = 1 has B = 0.135). B and M use gaps < 30 min; the think-time fit uses gaps < session_gap.
- The gating clock is feature.active (B01 runs before B11 in the registry).

- Eval round 2 (integration.md §10.2 #8): an active tick whose 6-h window holds no true gap yet (lone events of a session carried over from the previous tick) writes undefined (NaN) descriptors to `behavior.timing` instead of nothing, so the series is not stale while the entity is active.
- Open (cadence.md §17 item 1, integration.md §10.7 item 2): B24 blends B11's pm (a G test whose overdispersion is under-estimated) while the timing stratum holds < 64 entries; after a cadence switch to 60 s timing is one of the main single-tick drivers on control t-ticks.

## B12 — BeaconEngine [new]

- **File:** `backend/app/engines/behavior/beacon.py`
- **Layer / order / interval:** behavior / 12 / 1

**Purpose.** (Previously B13.) Detect C2 beaconing that is jitter-tolerant, additive or long-period, per (entity, rare destination), using valid tests on point events and population whitelisting.

**Algorithm.**

Input: act.rare_events event times and sizes (unit marks, not binned counts). Keep up to 256 per (e, dest) over up to 7 d, only for destinations with prevalence ≤ 20% that are not class-shared.
Evaluated when a new event arrives, at most once per 4 ticks per pair, once n ≥ 12:
1) Renewal-regularity test (primary). Fit a Gamma shape κ̂ to the intervals Δ by MLE (Minka's approximation plus Newton steps). LR = 2[ℓ(κ̂) − ℓ(1)], one-sided for κ̂ > 1.
   - The p-value comes from a precomputed Monte-Carlo null table per n (12..256, 1e5 simulations each, stored as constants in lib/evt.py), because the χ²₁ asymptotic is 2–5× anti-conservative at n ≤ 40 (measured).
   - Measured: ±30% uniform jitter at P = 300 s gives median p = 1.7e-8 at n = 12; a Poisson null gives fewer than 1e-3 below 1e-4.
2) Size constancy: a Gamma LRT on byte sizes against the destination class's size dispersion.
3) Strict-period test for low jitter: the Z²₂ (Buccheri) statistic on phases over a trial-period grid oversampled at step 1/(5T) (T = span), band [T/4, 10 s], with the effective-trials correction. Rayleigh at the best period is only weak under renewal jitter (median p = 0.14 at n = 40, measured), so it is not used alone.
4) Combine: p = min(3·min(p_renewal, p_size, p_Z²), 1) (Bonferroni).
Alert when p < 1e-5, the destination prevalence is ≤ 2 entities, and the class does not share (dest, P̂ ± 10%).
score.beacon = −log10 p. Accumulator, axis c2. RITA-style dispersion values are reported only as descriptors.

**Reads.** act.rare_events, model.vocab@__system__ (prevalence), model.class, model.feedback (allowlist)

**Writes.** model.beacon@(s,e), behavior.score[beacon], behavior.acc_alarm[beacon], profile.extra.beacons; event beacon

**Perf.** ≈0.1 ms per pair evaluation (fit, table lookup, Z² on ≤256 events × ≤512 trials); ≤2 ms per tick

**Unit test.** (a) One request every 300 s ± 30% for 20 events to a rare destination, added to 22 requests per tick of normal traffic: p < 1e-6.
(b) A health check to svc.corp.local shared by 3 class members: no event.
(c) Over 1000 Poisson-random destination streams: fewer than 0.5% have p < 1e-3 (validity of the MC table).
(d) Log-normal human gaps with σ = 1: fewer than 0.5% have p < 1e-4.

**Integration notes (as built, docs/lib3/integration.md).**
- Renewal p at the least-favourable κ0 = 1.5 of a composite null (max'ed with the exact κ = 1 table p); size constancy is a population rank test; the Z²₂ band is anchored at the median interval.
- Destination prevalence: m_vocab.dest_prevalence (B08) on the destination name, R2's decayed counts as the fallback.

## B13 — BudgetEngine [new]

- **File:** `backend/app/engines/behavior/budget.py`
- **Layer / order / interval:** behavior / 13 / 1

**Purpose.** (Previously B14.) Long-horizon cumulative budgets in DLP style, split by evidence axis: volume, exfil (novel destinations) and breadth (distinct objects, destinations and templates). They catch low-and-slow, duty-cycled and enumeration behaviour that never crosses a per-tick threshold.

**Algorithm.**

Quantities, in natural units:
- volume axis: bytes_up, bytes_down, write requests, active slots;
- exfil axis: bytes_up to novel destinations (first seen less than 7 d ago, or class-rare, from model.vocab), and DNS label bytes per new eTLD+1;
- breadth axis: distinct object ids per template (HLL p = 10, per day, from act.objs), distinct templates, distinct destinations.
Storage: hourly bins in a 28 d × 24 ring per quantity.
Horizons: 1 h, 8 h, day-to-date at the same local phase and day_type, and 7 d.
Threshold: POT/GPD with u = P98 of the same-phase history and PWM estimators (lib/evt.py).
- The evaluation quantile is q = 0.01/(#Q·#H·24) per hourly evaluation, so the family false-alarm rate is ≤ 0.01 per entity-day.
- With fewer than 20 days of history, use a peer-pooled GPD rescaled by the entity median.
- There is no upward drift term.
Alert when B > z_q AND B − median > abs_floor_Q (config, e.g. 20 MB or 500 objects) AND B/peer_median exceeds its own history P99 (a common-mode guard; not applied to exfil).
Scores (accumulators): score.budget_vol, score.budget_exfil, score.budget_breadth = −log10 of the tail p for each axis.
Emit budget_exceeded with natural-unit text. Aggregate per actor when model.link has chains. Honours rollback: the ring bins carry ts, and bins after rollback_to are recomputed from the held journal on release.

**Reads.** feature.nat, feature.tctx, act.objs, act.stream, act.rare_events, model.vocab, model.class, model.link, behavior.trust, model.control

**Writes.** model.budget@(s,e), behavior.score[budget_vol, budget_exfil, budget_breadth], behavior.acc_alarm[...], behavior.budget, behavior.axes, profile.extra.budget; event budget_exceeded

**Perf.** O(Q) per tick; GPD fits round-robin; ≤2 ms per tick

**Unit test.** (a) 20 days of daily upload ~ LogNormal(ln 50 MB, 0.3), then a day with 30% more on every tick: the day-to-date alarm fires by 14:00 local while every per-tick |z| < 2.
(b) 2000 distinct /orders/view ids in 8 h against a usual 40: budget_breadth alarm.
(c) The whole class at ×2.5 for a day: no budget_vol alarm (peer guard).
(d) 3 MB per day to a novel external destination: below abs_floor, so no alarm, but the value is recorded for class aggregation in B18.

**Integration notes (as built, docs/lib3/integration.md).**
- Reads also dns.qname_set (DNS label bytes). The common-mode guard adds a level-q test on log(B / peer median); thresholds are computed in log space; default absolute floors (config budget_abs_floor, contract I) include bytes_up 5 MB.
- Eval round 2 (integration.md §10): the log-scale floor of a COUNT quantity (writes, slots, objs, templates, dests) also covers its counting noise, sqrt(m + 1)/(m + 1 + unit) at level m (`budget.floor_scale(..., count_unit=unit)`). A phase whose count history is constant (0 writes at 03:00, 5 destinations in 7 d) had a log scale of 0 floored at 2 %, so one more count was a 20-50σ residual and a tail p of 1e-10 … 1e-38 on clean entities. Bytes quantities are unaffected.

## B14 — ChangepointEngine [upgrade]

- **File:** `backend/app/engines/behavior/changepoint.py`
- **Layer / order / interval:** behavior / 14 / 1

**Purpose.** (Previously B15; replaces cosine drift.) Sequential detection of persistent, intermittent and gradual shifts, measured against the reference anchor so a creeping baseline cannot cancel them. Uses wall-clock thresholds, estimates onset for rollback, and tests creep.

**Algorithm.**

Input: zr (residuals against the reference anchor) for the 12 KEY_FEATURES, prewhitened per feature with AR(1) φ from committed history clipped to [0, 0.8]: u = (zr_t − φ·zr_{t−1})/sqrt(1 − φ²). ψ = clip(u, −3, 3). NaN contributes 0 and does not reset.
(a) CUSUM bank: 12 features × 2 sides × k ∈ {0.25, 1.0} = 48 charts.
   - Family false-alarm target 0.02 per entity-day, so each chart's ARL = 2400 days.
   - h = seq.h_gauss(k, 2400·86400/Δt) from Siegmund's formula: at 900 s, h = 19.4 (k = 0.25) and 5.35 (k = 1); at 60 s, 24.8 and 6.7; at 3600 s, 16.6 and 4.66.
   - Expected run length at a 1σ shift ≈ 27 ticks at 900 s.
   - A round-robin audit bootstraps one entity per hour on committed residuals (block 30). If the realised rate exceeds 2× the target, that entity's h is multiplied by 1.1 (bounded at 1.5×).
   - p_eq = min(1, 48·exp(−2k(S + 0.583)))
(b) Crosier MCUSUM on Sinv_half·zr with k = 0.5, h from a precomputed Monte-Carlo table in lib/seq.py (d ≤ 16) at ARL = 100 days.
(c) BOCPD at the hourly scale on intensity z̄r and behavior.wh: Normal-Gamma, hazard 1/168 h, R_max = 336, pruned below 1e-4. cp.prob = P(r ≤ 3 h); onset = MAP run start.
(d) Creep, daily: Mann–Kendall on 14 daily values of (entity − class median) of the mean zr per group, compared with golden. Emit baseline_creep when p < 0.01 and |Sen slope| > 0.05 log-units per day.
All four are accumulators.
- behavior.cp.onset τ̂ = the last tick at which the alarmed statistic was 0.
- Axes are the groups of the contributing features.
- Cumulative excess is reported in natural units.
- Statistics reset after 2(t − τ̂) clean ticks.
- The CUSUM state vector is stored for replay.

**Reads.** behavior.zr, behavior.wh, model.density, model.baseline (golden), model.class, behavior.trust, behavior.quarantine, model.control

**Writes.** model.cp@(s,e), behavior.score[cusum, mcusum, bocpd, creep], behavior.acc_alarm[...], behavior.cp.prob, behavior.cp.onset, behavior.cusum_state, behavior.axes, profile.extra.regime.delta_by_feature; event baseline_creep

**Perf.** ≈48 scalar updates plus one d×d mat-vec per tick; BOCPD hourly; ≤3 ms per tick

**Unit test.** (a) N(0,1) at Δt = 900 for 60 simulated days: the family alarm rate is ≤ 0.03 per day across 50 seeds. The same test at Δt = 60 gives a rate within [0.5, 2]× of that.
(b) A +1σ shift from t = 500 at Δt = 900: alarm within 60 ticks, with τ̂ within ±20.
(c) Duty 30% × 3σ: alarm within 30 ticks.
(d) Alternate-tick 3σ: alarm within 10 ticks.
(e) An entity ramping +0.2σ per day while the class is flat for 14 d: baseline_creep.
(f) The current anchor poisoned +1σ while the reference is unchanged: the input zr still reflects the shift.

**Integration notes (as built, docs/lib3/integration.md).**
- Chart inputs are the key zr of features the entity's OWN baseline identifies at the bucket (m_baseline.own_support: ≥ 60 weighted minutes of own rows and ≥ 2 rows of the feature); other inputs are NaN (0 increment, no reset). Residuals against a pure backoff / hyperprior predictive carry a systematic offset (e.g. z ≈ −2 for a quiet 4xx rate under Beta(0.5, 0.5)) that would alarm within hours.
- The CUSUM / MCUSUM statistics restart at 0 on the first live tick after warm-up (they ran against a model that was being learnt from those very rows).

- Canonical grain mode, end of warm-up (eval round 3, integration.md §10.3): the charts restart on the first live tick whatever its type. The restart used to run on the first live H decision tick only, and `_hold_latches` re-emitted a warm-up latch as a live accumulator alarm on the live ticks before it. Test: test_b14_changepoint.py::test_canonical_first_live_tick_between_h_ticks_restarts_the_charts.

**Canonical grain mode (cadence.md §7.2, §17).** The charts step on H decision rows only (decimated, non-overlapping sampling) and their ARLs are counted in hours, so the thresholds do not depend on Δt: cusum h = 16.60 (k = 0.25, ARL 2400 d) and mcusum h = 22.72 (d = 12, ARL 100 d) at every cadence. φ is learnt on H lag-1 pairs; BOCPD takes the hourly zr intensity mean and `wh` directly; creep uses daily means of H rows. Between H ticks B14 re-emits a latched cusum / mcusum / bocpd / creep alarm on every Q / T tick until the next H tick decides (otherwise B25, B27 and the eval saw an on-off train: 71 spurious "change"-path onsets on pack A). The CUSUM state ring keeps 96 H states for B29's replay.

## B15 — IdentityModelEngine [upgrade]

- **File:** `backend/app/engines/behavior/identity_model.py`
- **Layer / order / interval:** behavior / 15 / 1 (window collection every tick, every H tick in canonical mode; fit every 96 ticks or 24 h)

**Purpose.** (Previously B4; replaces fingerprint.py separability.) Measure how identifiable each IP and each class is, with honest blocked cross-validation on ABSOLUTE representations. Report what distinguishes each one and whom it is confused with, and fit the calibration of each modality's LLR for attribution. This engine outputs no p-values; B24 owns them.

**Algorithm.**

Runs every 96 ticks or 24 h, whichever comes first, with a random phase per system.
Windows: K = 4 committed active ticks, non-overlapping, kept in model.idwin (at most 3000 per system).
Window vector (absolute, never behavior.z):
- the median of feature.vec, with NaN replaced by the role-class bucket median and a missing mask appended;
- the IQR of 8 key vec features;
- the mean feature.sketch (80);
- behavior.timing (B, M, log think_mu);
- sin/cos of the local hour at the window centre, and a weekday flag.
PCA to d = 48 per system, fitted on pooled windows.
Model: OAS pooled within-entity covariance with an eigenvalue floor, WCCN within role, and LDA (solver lsqr, shrinkage auto).
Blocked CV: 3 contiguous folds per entity with a 2-window gap. Modality-drop importance is computed only on every 4th run.
Per entity:
- recall@1 and recall@K;
- EER_hard: genuine windows against held-out windows of the 3 nearest impostors by Bhattacharyya distance;
- T99; the confusion row; anonymity sets by union-find over pairs with confusion > 0.2.
- separability = clip(1 − 2·EER_hard, 0, 1).
Modality LLR calibration a_m, b_m: logistic regression of genuine versus impostor CV scores for m ∈ {gauss, vocab, rhythm, seq, client, timing}. The non-gauss modality LLRs are computed with the candidates' own models through pure helpers (ppm.loglik, a multinomial vocab log-likelihood, a Beta rhythm log-likelihood, stack Dirichlet, a gap-histogram log-likelihood).
Class identifiability: the same procedure with role labels.
Distinctive traits: Fisher scores in natural units against role peers, plus Monroe–Colaresi–Quinn log-odds on the vocab.
Emit low_identifiability (INFO) when EER_hard > 0.2.

**Reads.** feature.vec, feature.sketch, feature.tctx, feature.active, behavior.trust, behavior.timing, model.vocab, model.rhythm, model.seq, model.client, model.timing, model.class, model.control

**Writes.** model.identity@(s,__system__), model.idwin@(s,__system__), profile.separability, profile.extra.identity, class profiles' identifiability; event low_identifiability

**Perf.** LDA at 600×64×20 = 29 ms (measured). About 4 fits plus 9/4 amortised drop fits ≈ 180 ms per system per run, ≈6 ms per tick amortised over 3 systems at a 96-tick stride; ≈1 s in a 180-tick warm-up.

**Unit test.** Build 6 synthetic entities in ABSOLUTE feature.vec space: 4 whose means are 1.5 within-sd apart in 5 dimensions and have distinct sketches, plus 2 that are identical.
- recall@4 ≥ 0.95 and EER_hard < 0.05 for the 4 distinct entities.
- The identical pair has EER_hard > 0.3, lists each other in confusable_with, and has separability < 0.4.
- A static check asserts that consumes excludes behavior.z and behavior.zi.

**Integration notes (as built, docs/lib3/integration.md).**
- Reads also behavior.trust / trust_prov / quarantine, model.link, act.tokens, act.stream, act.stream_frac, client.stack_set, tls.sni_etld1_set, dns.qname_etld1_set and l4.dport_set (live modality calibration). Interval 1 (window collection) with the fit every 96 ticks or 24 h. EER_hard is the max pairwise EER over the 3 nearest impostors. model.identity fields: contract C.
- Live modality calibration opens an entity's next calibration window at least CAL_GAP_S = 4 h after its previous one (≈ 6 windows per entity per day, MIN_CAL still reached within hours): the PPM / vocab LLRs under 4 candidates cost ~7 ms per system per tick when three entities sampled continuously.
- The window vector zeroes the client.stack_set block of the mean sketch (R20.2): client stacks are their own capped modality in B16.

## B16 — AttributionEngine [new]

- **File:** `backend/app/engines/behavior/attribution.py`
- **Layer / order / interval:** behavior / 16 / 1

**Purpose.** (Previously B16.) Continuous open-set identity attribution. For each window, decide whether it is the IP itself, another known IP or class (impersonation, IP swap, role takeover), or nobody known. Candidates are scored with their own models on absolute data.

**Algorithm.**

Window: the last K = 4 active ticks. Idle ticks abstain; only the presence term applies.
Candidates: self, the top 5 by LDA score in model.identity space (absolute window vector), the own role class, and the top other role.
Modality log-likelihoods, each under the candidate's model:
- Gaussian block in LDA space;
- vocab multinomial with Dirichlet smoothing toward the system, scaled by min(n_tok, 20)/20;
- rhythm presence under model.rhythm;
- PPM log-likelihood via ppm.loglik, computed at window completion;
- client stack under model.client;
- gap histogram under model.timing.
Each modality is calibrated by a_m·llr + b_m and capped at ±4 nats per window, so a browser upgrade alone cannot flip identity.
Posterior over candidates plus an explicit unknown hypothesis (prior 0.05).
score.identity = −log10 of the self posterior's LLR tail. B24 calibrates it with strata (daypart, regime). Instantaneous, axis identity.
Other-identity CUSUM:
- λ = clip(max_{j≠e} l_j − l_e, −4, 8); S = max(0, S + λ − 0.5).
- h from seq.h_for with ARL = 100 days, counted in windows per day as estimated for the entity.
- Emit identity_mismatch when S ≥ h, the same j* wins in at least 3 of 4 windows, π_j* ≥ 0.9, and the CV confusion(e, j*) < 0.1. Otherwise downgrade to INFO.
- Severity is HIGH when j* is active elsewhere at the same time (impersonation), otherwise MEDIUM.
Unknown CUSUM: +1 when max_j p_j < 0.01, else −0.5; alarm at 2.
- HIGH when the class typicality p < 0.01 ('unlike anyone');
- MEDIUM for 'same class, different individual';
- INFO for a young IP.
The common-mode flag never suppresses identity. When the entity has shared_ip, events are downgraded to INFO.

**Reads.** feature.vec, feature.sketch, feature.active, act.tokens, act.stream, client.stack_set, behavior.timing, model.identity, model.vocab, model.rhythm, model.seq, model.client, model.timing, model.class, model.link, profile.extra.continuity

**Writes.** behavior.score[identity], behavior.axes, behavior.id, profile.extra.attribution; events identity_mismatch, unknown_identity

**Perf.** Cheap modalities every tick (≈30 µs per candidate); PPM and gap terms once per window; ≤10 ms per tick at 40 entities

**Unit test.** (a) Fit two distinct entities A and B (store their models with put_model). Feed A's absolute rows and tokens under B's key for 6 active ticks: identity_mismatch on B with looks_like = A and posterior ≥ 0.9; HIGH if A is also active.
(b) Feed B its own rows: no event over 200 ticks.
(c) Rows from an unseen distribution: unknown_identity.
(d) Change only the client stack: no mismatch (cap).
(e) A static check asserts that consumes excludes behavior.z.

**Integration notes (as built, docs/lib3/integration.md).**
- The unknown CUSUM only climbs on full K-row windows (partial windows of a new or long-silent entity only let it decay): the chi² typicality is calibrated on B15's K-row windows, and a 1–2 row window (IQR 0, noisier median) failed it for enrolled entities on their first ticks.
- The per-candidate typicality is calibrated by the candidate's held-out genuine T99 from B15 (d² × chi2_r⁻¹(0.99)/T99 before the chi2 tail), so p < 0.01 means farther than 99 % of the entity's own held-out windows; the raw chi2_r assumed a within-entity covariance of I, which WCCN gives only on average.

**Canonical grain mode (cadence.md §8, §7.2).** A window is 4 active H rows (`m_identity.grain_row`: `feature.vec.h`, `feature.sketch.h`, timing descriptors and clock features at the row's midpoint tctx), i.e. 4 h of data at every cadence. The instantaneous `identity` score is marked `overlap` (consecutive windows share 3 rows): it takes the single-tick path only and never enters the H evidence CUSUM. The other-identity and unknown CUSUMs step only on windows that share no row with the previous step; h uses windows per day = active H rows per day / 4.

## B17 — EntityLinkEngine [new]

- **File:** `backend/app/engines/behavior/entity_link.py`
- **Layer / order / interval:** behavior / 17 / 1

**Purpose.** (Previously B17.) Keep identity continuous across DHCP and VPN re-addressing, flag takeovers dressed up as swaps, detect shared or NAT IPs, and chain IP-hopping actors.

**Algorithm.**

Trigger: the appearance of a new entity B (store.first_seen within the last 24 h and fewer than 24 committed active ticks). It is evaluated on each of B's first 8 active ticks.
Candidates A: same system and same static pool or DHCP scope, with last_seen(A) ≤ first_seen(B) + Δt, i.e. no activity of A after B appears. A's dormancy is not required at trigger time; A's continued silence is required only to confirm the link.
Fellegi–Sunter log-odds LO:
- behavioural LLR = Σ_windows[l_A(x) − logsumexp_{j≠A, incl. new} l_j(x)], using B16's calibrated, capped modality helpers;
- vocab MNB LLR against the system background;
- device LLR = ln P(same stack) − ln(stack frequency in the system);
- topology prior: +2 for the same /24 or DHCP scope, else −2;
- time prior: exponential in the gap, with rate 1/lease.
Resolve with a Hungarian one-to-one assignment. Link when LO ≥ 5 with a margin ≥ 2 over the runner-up (expected within 4 active ticks of B).
On link:
- write model.link (version + 1) and continuity {continuity_id, aliases, linked_from};
- mark A identity_moved;
- every learner then seeds B := B_own + 0.5·A on the tick it sees the new version (contract H covers baseline, rhythm, vocab, client, seq, timing, budget, density, calibration rings and identity), so a link that arrives ticks later still seeds correctly.
If behaviour matches (LO ≥ 5) but the device LR ≤ 0.1, emit possible_impersonation instead of linking. If A becomes active while B is active, retract the link (link_retracted), and learners undo the seed through rollback_to = the link time.
Shared IP, every 32 ticks or 8 h: fit GaussianMixture with k = 1 and k = 2 (diagonal) on the last 96 active zi. Emit shared_ip when all of the following hold:
- BIC(2) < BIC(1) − 10 on 3 consecutive runs;
- the components co-occur in the same hours;
- two disjoint stacks overlap in client.stack_events.
Then set entity_kind = 'ip-class'.
Actor chains: links within 24 h form model.link.actors, which B13 uses.

**Reads.** store.first_seen/last_seen, feature.active, feature.vec, feature.sketch, act.tokens, client.stack_events, behavior.zi, model.identity, model.vocab, model.client, model.rhythm, model.seq, model.timing, model.class, ctx.config.dhcp_scopes

**Writes.** model.link@(s,__system__), profile.extra.continuity; events entity_resolution (conf = σ(LO − 5)), possible_impersonation, shared_ip, identity_moved, link_retracted

**Perf.** Driven by triggers; GMM ≈2 ms per entity per run; ≤1 ms per tick

**Unit test.** (a) A goes silent at t, and B appears at t + Δt with A's persona and stack: entity_resolution within 4 active ticks of B, conf ≥ 0.9. A new C with a different persona appearing at the same time is not linked.
(b) B with A's behaviour but a different stack and TTL: possible_impersonation.
(c) A reactivates: link_retracted, and B's baseline equals its unseeded fit.
(d) An IP with two personas on disjoint, overlapping stacks: shared_ip after 3 runs.

**Integration notes (as built, docs/lib3/integration.md).**
- behavior.zi is read only by the shared-IP test (linking is absolute). Shared-IP fit: PCA(≤ 4) + deterministic 2-component EM, first run at ≥ 64 rows. model.link fields: contract C; accessor lib/m_link.
- A retracted link is undone by B28 (model.control rollback_to = t_link, release [t_link, now] on the seeded entity).

**Canonical grain mode.** B17 runs on H ticks and scores the same 4-active-H-row windows as B16 (cadence.md §8, §17).

## B18 — ClassMonitorEngine [new]

- **File:** `backend/app/engines/behavior/class_monitor.py`
- **Layer / order / interval:** behavior / 18 / 1

**Purpose.** NEW: class-as-entity deviation detection. Every dynamic role class, static CIDR class and pool class gets an aggregate hour-of-week profile, calibrated class-level detectors, class incidents, class risk and a portrait. This covers a compromised subnet or pool, or a class-wide behaviour change, which per-entity views discount as common mode.

**Algorithm.**

Classes per system s:
- role classes with at least 2 members in s ('class:<rid>');
- static CIDR classes with at least 3 members ('class:static:<name>');
- pools ('class:pool:<cidr>').
Members are those with membership probability ≥ 0.5.
1) Aggregate row per tick, built from members' feature.nat, feature.expo, feature.active and tokens:
   - summed per-minute counts with summed exposure;
   - pooled ratios Σk/Σn;
   - exposure-weighted averages;
   - active fraction m_act/m;
   - union token counts;
   - count of new external destinations.
   Written to behavior.class.agg.
2) Model model.classagg: the same conjugate seasonal helpers as B03 (current and reference anchors, bin48/bin168, hyperpriors), committed with class trust from B28 at the class key. It honours model.control@(s, class:<id>).
3) Scores at the class pseudo-entity:
   - class_int: volume group, NB/Student-t on the aggregate; instantaneous.
   - class_shape: composition and ratio groups, plus the JSD of pooled templates against the class profile; instantaneous.
   - class_rhythm: a Beta-binomial CUSUM on the number of active members per 15-minute slot, with a wall-clock threshold; accumulator.
   - class_novel: over new values v adopted by m_v ≥ 2 members in 24 h, score = Σ w_v·(−log10 P(M ≥ m_v | the class's historical adoption rate)). Weights: w_v = 3 if external, ×3 if upload-dominant, ×3 if sensitive, ×2 if a new eTLD+1 org-wide; w_v = 0.1 for an internal read-only template. Calibrated over the class history; accumulator; axes exfil, c2 or categorical.
   - class_coherence: a binomial tail on the number of members with p < 0.01 in the same direction on the same group, against an expected 0.01 per member; instantaneous.
4) Coherent intensity-only and system-wide app-error or transport shifts emit coherent_shift or class_shift (INFO/LOW). Risky adoptions emit class_adoption_risky.
5) The class profile gets extra.class_monitor {aggregate p5/p50/p95, active fraction by bin, adoption history}.

**Reads.** feature.nat, feature.expo, feature.active, feature.tctx for members; act.tokens; model.class; model.vocab@class (adoption records); behavior.common.*; behavior.z (member directions); behavior.trust@class; model.control@class

**Writes.** model.classagg@(s,class:<id>), behavior.class, behavior.score[class_*]@class:<id>, behavior.pm@class, behavior.axes@class, profile(class).extra.class_monitor; events coherent_shift, class_shift, class_adoption_risky

**Perf.** About 12 classes × 0.2 ms; ≤3 ms per tick

**Unit test.** (a) A 5-member class at ×2.5 volume with the same mix: class_int p < 1e-6, but after B25 the class severity is capped at LOW and there are no member incidents.
(b) 4 of 6 members start uploading 30 KB per 15 min each to one new external SNI. Each member's p > 0.01, while class_novel p < 1e-6 gives a class incident ≥ MEDIUM within 4 ticks of the 3rd adopter.
(c) A static CIDR class of 3 IPs gets its own model.classagg and portrait.
(d) 6 of 8 members adopt an internal /v2/orders: class_novel p > 0.01.

**Integration notes (as built, docs/lib3/integration.md).**
- Perf: the exact m_baseline predictives cost ~0.4 ms per midp call, so budget ~1.5–2 ms per class-tick rather than 0.2 ms.
- class_rhythm in canonical grain mode (eval round 3, integration.md §10.3). (1) A member is active in a 15-min slot only if its act.slot_events has an event IN that slot; every slot a tick overlaps is registered with the present-member count, and every slot complete at the tick is scored and learned in time order (a 3600-s tick closes four). The v2 path counted a member active in a tick's slot when it was active anywhere in the tick, so an hourly warm-up taught the hour's activity as one slot's (4 members each active in a different quarter: 4 of 4 per slot at 3600 s, 1 of 4 at 900 s) and the 900-s live slots then read as a class-wide drop (pack B: erp-prod class:r3 S_lo latched Sat 20:15 → Tue 23:15). (2) A slot is scored only when its own bin (local hour × day type) holds ≥ 3 decayed slots (RHYTHM_MIN_BIN_W, most of one observed day); an empty bin used to fall back to the pooled prior of strength 4, i.e. to the other bins: a human class formed during a weekend 900-s phase scored its Monday-morning slots against weekend activity (z at the clip, S_hi = 21 by the first live tick, HIGH temporal class incident in scripts/smoke.py at Mon 09:13). The observation is learned either way. Tick mode keeps the v2 path (golden reference). Tests: tests/engines/test_b18_rhythm_grains.py.

## B19 — MixtureEngine [new]

- **File:** `backend/app/engines/behavior/mixture.py`
- **Layer / order / interval:** behavior / 19 / 32
- **Status:** not built (P2). The module does not exist and the engine is not registered; its detector keeps its slot in `lib/detectors.py` (`P2_DETECTORS`) and is never scored (B25 treats it as not scored, not as degraded). Enable only after the ablation gate (eval.md gate 13); integration.md §10.6 lists the evidence so far.

**Purpose.** (Previously B18; P2.) A density model for entities whose residual behaviour stays multimodal after hour conditioning (burst/idle jobs), so that values between the modes score as unlikely.

**Algorithm.**

Gate: bimodality coefficient BC = (g² + 1)/(κ + 3(n−1)²/((n−2)(n−3))) > 0.555 on the first 2 robust PCs of committed zi.
Fit: GaussianMixture with k = 1..3 on PCA-reduced zi (d ≤ 6), full covariance, reg_covar = 1e-3, warm start. Select by BIC, with ΔBIC > 10 required.
Score every tick: score.mixture = −logsumexp_j[log π_j + log N(z; μ_j, Σ_j)]. Instantaneous, axis shape.
Honours model.control. Runs every 32 ticks or 8 h.

**Reads.** behavior.zi, model.density, behavior.trust, model.control

**Writes.** model.mixture@(s,e), behavior.score[mixture]

**Perf.** ≈80 ms per fit on gated entities only; ≤2 ms per tick amortised

**Unit test.** Train on two modes at ±3. A point at 0 gives p < 1e-3 while its univariate robust z ≈ 0. A unimodal entity is not fitted.

## B20 — SessionProfileEngine [new]

- **File:** `backend/app/engines/behavior/session_profile.py`
- **Layer / order / interval:** behavior / 20 / 1
- **Status:** not built (P2). The module does not exist and the engine is not registered; its detector keeps its slot in `lib/detectors.py` (`P2_DETECTORS`) and is never scored (B25 treats it as not scored, not as degraded). Enable only after the ablation gate (eval.md gate 13); integration.md §10.6 lists the evidence so far.

**Purpose.** (Previously B19; P2.) A session-level shape profile, to catch scripted scraping, marathon sessions, and atypical depth or write mix.

**Algorithm.**

Each completed session (boundaries from model.seq session_gap) becomes a 12-dim vector:
- log duration, log number of actions, number of templates, write ratio, 4xx ratio;
- log bytes up and down;
- sin/cos of the start hour, weekend flag;
- mean think time, first-token IDF.
Model per entity, refit every 16 sessions: MinCovDet on the last 200 sessions, shrunk toward the class covariance with ν = 20.
score.session = −log10 p of the robust d². Instantaneous, axis sequence.

**Reads.** act.stream, model.seq, model.vocab, model.control

**Writes.** model.session@(s,e), behavior.score[session], profile.extra.session_typical

**Perf.** ≤1 ms per tick

**Unit test.** Train on 150 human sessions of 5–30 min with gaps of about 8 s. A 6-h session with 0.8-s gaps gives p < 1e-3.

## B21 — CrossSystemEngine [new]

- **File:** `backend/app/engines/behavior/cross_system.py`
- **Layer / order / interval:** behavior / 21 / 4
- **Status:** not built (P2). The module does not exist and the engine is not registered; its detector keeps its slot in `lib/detectors.py` (`P2_DETECTORS`) and is never scored (B25 treats it as not scored, not as degraded). Enable only after the ablation gate (eval.md gate 13); integration.md §10.6 lists the evidence so far.

**Purpose.** (Previously B20; P2.) An IP-centric view across business systems: first access to a system, and lateral spread.

**Algorithm.**

Keyed by IP across store.systems(), with decayed counts of (IP, system) pairs.
- Novelty: three-tier Dirichlet (IP → static class or role → org) with Good–Turing maturity.
- Lateral spread: distinct systems per 24 h under an NB predictive with a class prior.
- Support for a new bipartite edge: Adamic–Adar from peers.
score.cross_system = −log10 p. Instantaneous, axis discovery (stage lateral).
Emit first_access_system when the tier ≥ class and p ≤ 0.005.

**Reads.** feature.active across systems, model.class, ctx.config.ip_classes

**Writes.** model.xsys@(__org__,__org__), behavior.score[cross_system] (per system|ip), profile.extra.systems_accessed

**Perf.** Less than 1 ms per tick

**Unit test.** An IP active only in erp-prod for 30 d starts using oa-portal /admin: event at class tier. The same access adopted class-wide is suppressed.

## B22 — ActionEmbeddingEngine [new]

- **File:** `backend/app/engines/behavior/action_embedding.py`
- **Layer / order / interval:** behavior / 22 / 256
- **Status:** not built (P2). The module does not exist and the engine is not registered; its detector keeps its slot in `lib/detectors.py` (`P2_DETECTORS`) and is never scored (B25 treats it as not scored, not as degraded). Enable only after the ablation gate (eval.md gate 13); integration.md §10.6 lists the evidence so far.

**Purpose.** NEW (P2, optional learned modality). Learn dense semantic embeddings of templated actions and entities from co-occurrence, with no GPU. Enabled only if the ablation gate shows a gain for attribution, role clustering or semantic novelty.

**Algorithm.**

Runs every 256 ticks or 24 h per system, on committed act.stream data.
1) Template co-occurrence within sessions (window ±2 tokens) → PPMI matrix with context-distribution smoothing α = 0.75 → randomized truncated SVD with k = 16, E = U·Σ^0.5. This is equivalent to SGNS word2vec (Levy & Goldberg 2014) but deterministic and cheap.
2) Entity × template Poisson NMF (KL, 4 components) for a collaborative p̂(e, v).
Consumers, all through the store:
- B16 adds a Gaussian block on session embeddings (behavior.embed.session = the mean of E over the window's tokens);
- B02 can use the cosine between template-family centroids;
- B08 adds a semantic-novelty bonus, 1 − max cosine between a new template and the entity's used templates.
Every consumer must degrade gracefully when model.embed is absent.

**Reads.** act.stream, model.template, behavior.trust

**Writes.** model.embed@(s,__system__), behavior.embed.session

**Perf.** Randomized SVD on a 4000×4000 sparse PPMI matrix ≈50 ms per run; ≤0.5 ms per tick amortised

**Unit test.** Build a synthetic grammar with two disjoint workflows of 5 templates each. Within-workflow cosine > 0.8 and cross-workflow cosine < 0.2. NMF reconstructs held-out entity-template counts with lower KL than the unigram.

## B23 — FeedbackEngine [new]

- **File:** `backend/app/engines/behavior/feedback.py`
- **Layer / order / interval:** behavior / 23 / 1

**Purpose.** (Previously B21.) The analyst loop: learn from verdicts, suppress known-benign patterns, accept legitimate change, re-weight detector families, hold the alert budget, and choose which cases to ask about.

**Algorithm.**

Reads new Labels via store.labels(since).
1) Precision per (family, contributor key) ~ Beta(1 + TP, 1 + FP). Risk multiplier π = clip(E[prec]/0.5, 0.2, 2).
2) Once there are at least 20 labelled incidents: stacking logistic regression on [min(8, log10(1/e_day(p_family)))…, stage count, signature max severity], with an L2 penalty toward uniform (λ = 5). family_w = softplus(coef), normalised. Isotonic calibration of P(malicious) once there are at least 100 labels.
3) Pattern policies from fp or benign_known labels with scope = pattern:
   - tokens σ = {kind, top-5 (feature, sign), new categorical tokens};
   - max level = the label's e_day ÷ 10, i.e. one order of magnitude of headroom;
   - TTL 14 d.
   An incident is suppressed iff the Jaccard similarity is ≥ 0.6 and it is within that level. It is stored with status = suppressed and still contributes to risk at 0.25.
4) benign_known adds allowlist entries (entity or class, dim, value), used by B08 and B12.
5) expected_change goes to accept[(s, e, ts)]; tp or malicious goes to freeze[(s, e, ts)]. B28 consumes both.
6) Alert budget per system, updated daily: alpha_mult ← alpha_mult·exp(0.1·(target − observed)), clipped to [0.25, 4]. It rescales the e_day severity thresholds, never the calibration.
7) Label queue, at most 5 per day: 80% the highest-risk unlabelled cases, 20% the most uncertain |P − 0.5|, plus every incident held for more than 14 d.

**Reads.** store.labels, store.incidents, behavior.p_family, behavior.e_day

**Writes.** model.feedback@(__org__,__org__) {family_w, detector_prec, policies, allowlist, alpha_mult, accept, freeze, queue}

**Perf.** Less than 1 ms per tick

**Unit test.** (a) Label an incident fp with scope pattern. A new identical incident at the same level is suppressed. One that is 1.5 orders of magnitude rarer is not. One of a different kind is not.
(b) 30 synthetic labels where the sequence family is always fp: its weight falls below 0.5 × uniform.
(c) An expected_change label shows up in model.feedback.accept within 1 tick.

## B24 — CalibrationEngine [new]

- **File:** `backend/app/engines/behavior/calibration.py`
- **Layer / order / interval:** behavior / 24 / 1

**Purpose.** (Previously B22.) Turn every raw detector score into a per-entity p-value that is valid in finite samples, meaningful in the far tail, free of ties, stratified by daypart and cadence, and gated against poisoning. This is the single owner of detector p-values, including identity.

**Algorithm.**

Rings: for each (entity or class key, detector d, stratum) keep a sorted ring C of up to M = 256 scores, each with its ts.
- stratum = daypart(4) × cadence class. For identity, stratum = (daypart, regime tercile).
- Admission: committed ticks only (same D and trust gating as the learners), entity maturity n_eff ≥ 48, not quarantined.
- Ring entries after model.control.rollback_to are deleted; the ring is reset on a version change.
p-value: randomized conformal
  p = (#{c > s} + U·(#{c == s} + 1))/(|C| + 1),
with U ~ U(0,1) seeded by hash(s, e, d, ts) for reproducibility. This removes the point mass at p = 1 from sparse or accumulator detectors (silence, beacon, offhours, novelty, CUSUM at 0).
Tail: when s > u = q_0.90(C) and there are at least 10 exceedances, use a PWM-GPD fit (ξ ∈ [−0.5, 0.5], refit every 16 ticks): p = (N_u/N)(1 + ξ(s − u)/σ)^(−1/ξ).
Small samples (|C| < 64, e.g. a new stratum after a cadence switch): logit-blend the conformal p with behavior.pm[d] when present (model p-values are exposure-exact, so they do not depend on cadence); for accumulators use the seq stationary-tail p_eq; otherwise use the class-pooled ring. Weight n/(n + 64).
Health: KS of the randomized p on trusted ticks per detector per system, and the realised exceedance rate at e_day ≤ 0.03. When KS D > 0.05 or the rate ratio falls outside [0.5, 2], write weight_mult = 0.5 into behavior.calib_health.
Degraded inputs (NaN score) give NaN p, never 1.

**Reads.** behavior.score, behavior.pm, behavior.trust, behavior.quarantine, feature.tctx, behavior.regime, model.control, model.class

**Writes.** behavior.p, model.calib@(s,e) (detector rings), profile.extra.calibration, behavior.calib_health@(s,__system__)

**Perf.** 40 keys × 31 detectors × bisect ≈2 ms; GPD refits ≈1 ms per tick amortised

**Unit test.** (a) 2000 null Exp(1) scores: KS D < 0.03.
(b) A sparse detector with 90% zeros: randomized p has KS D < 0.03, while the deterministic version (control) gives D > 0.5.
(c) A score 10σ into a GPD tail with ξ = 0.2 gives a p within 2× of the true tail and below 1/(M + 1).
(d) Night scores never use the day ring.
(e) After a switch from 900 s to 60 s, the first 64 ticks blend with pm.
(f) Entries after rollback_to are deleted.
(g) NaN in gives NaN out.

**Integration notes (as built, docs/lib3/integration.md).**
- Small-sample prior order: the entity's own rings of the OTHER dayparts at the same cadence (pooled, ≥ 64 entries), then the pm prior, then the class-pooled ring. The first workday after a weekend warm-up, or the first night, is a new stratum for every detector, and several pm are only approximately calibrated.
- pm prior (W7 tuning, m_calib.pm_prior): pm is a p-value with two atoms — an accumulator's stationary p_eq is exactly 1 at a zero statistic (creep KS D was 1.0: the blend re-created a point mass at p = 1), and some entities' pm sits at the float floor on every null tick (a sparse nightly host scored against its peers: p at the floor every night). The prior randomises a tie at an atom with the mid-p rule, pm = 1 → 1 − π1 + u·π1 and pm at the floor → u·π0 (u the score's seeded U), π the atom's share of the entity's admitted pm history (ring '<d>@pm|<cc>', canonical H / Q detectors '<d>@pm|g:<g>'); below 16 entries π1 is the Laplace estimate and an unseen floor keeps its evidence. The body of pm is used as is. (Calibrating the whole of pm on that ring was measured and rejected: integration.md §8.2.)
- Live commits are capped by the governor's row-evidence weight (m_governor.evidence_weight, behavior.trust_evidence): it binds only for released rows (trust_prov has no accumulator factor). Warm-up commits keep trust 1.
- In canonical mode a Q score on the provisional transfer (behavior.prov < 0.5) is marked behavior.degraded 'provisional:q_transfer'.
- A model.control version change is detected against the gate's own version (as B25), so a bare 0 → 2 change resets the rings.
- Identity rings are stratified by (daypart, regime tercile, cadence class): B16's windows are K active ticks, so the identity score's null depends on the cadence like every other detector's (architecture §6). Key 'daypart|r<k>|<cc>'.
- Own-history floor (eval round 2, integration.md §10; m_calib.p_value, shared with B29 / B25 meta): below 64 entries the prior may add resolution BEYOND the ring but never contradict it — p ≥ #{ring > s}/(n + 1). A sparse entity (a nightly backup host: one scored H row a night, ~10 entries after two weeks) is scored every night against its peers with pm ≈ 1e-18 while its own ring holds the same score every night; the blend (weight n/(n+64) ≈ 0.14) issued p ≈ 1e-16 every night (MEDIUM evidence-CUSUM incidents on the L15 automation hosts).

**Canonical grain mode (cadence.md §9.1).** Strata per stream: H detectors (daypart, 'h'); Q detectors (daypart, 'q', prov); identity (daypart, tercile, 'h'); T detectors (daypart, cc) as above. The daypart of an H / Q score comes from the row's midpoint tctx. H rings fill at 24 entries a day at every cadence and are neither reset nor thinned by a cadence switch; admission is at the detector's decision ticks only (NaN elsewhere is skipped, not degraded). Health rates use p·86400/period_s(d). The pm ring of an H / Q detector is `<d>@pm|g:<g>`. B29 replays issued p through `m_calib.p_replay(grain, prov)`.

## B25 — FusionEngine [new]

- **File:** `backend/app/engines/behavior/fusion.py`
- **Layer / order / interval:** behavior / 25 / 1

**Purpose.** (Previously B23.) Combine dependent calibrated evidence into per-entity (and per-class) decisions with wall-clock false-alarm budgets. Uses weighted HMP, which is robust to p ≈ 1; re-calibrates the combination per entity; runs an evidence CUSUM only over instantaneous detectors; gives accumulators their own alarms; and applies axis-based reading rules and corroborated severities.

**Algorithm.**

1) p_family = wHMP(p_d, d in the family) = Σw/Σ(w/p), over non-NaN inputs only. A family with no valid inputs is marked degraded.
   p_inst = wHMP over families that contain instantaneous detectors (instantaneous members only); p_all = wHMP over all families.
   Weights w_f = family_w (feedback) × weight_mult (calibration health).
   Measured: HMP([1e-3, 0.999, 0.5]) = 0.003, where ACAT gives 0.5. Under ρ = 0.64 dependence the rate ratio is 1.095 at 1e-3.
2) Meta-calibration: per entity, rings of s = −log10 p_inst and −log10 p_all (strata daypart × cadence, randomized, GPD tail, owned here under model.calib meta keys), giving q_inst and q_all.
   e_day = q_all·86400/Δt.
   - Admission (W7 tuning): weight = min(gate weight, the governor's row-evidence weight behavior.trust_evidence), which binds for released rows (trust_prov has no accumulator factor); warm-up rows are admitted with trust 1 as before (gating them on their own evidence truncates the null: integration.md §8.2).
   - Tail (W7 tuning): the exceedances are winsorised before the PWM fit — values beyond the 99.9 % bound of the maximum of n_u exponential excesses (scale from ranks) are replaced by their expected order statistics (calib.winsorise_exceedances). −log10 of a valid p has an exponential tail, so a heavier fitted tail can only come from contamination (warm-up p_all of 1e-21 … 1e-37 gave ξ ≈ 0.3 and saturated q_all near 5e-4).
3) Evidence CUSUM: e_t = −ln q_inst; S = max(0, S + e_t − 3).
   - Alarm at S ≥ h = (ln ARL − 3.07)/0.94, with ARL = 33·86400/Δt. That is h = 5.31 at 900 s, 8.19 at 60 s and 3.83 at 3600 s (from the exact integral-equation ARL, ARL ≈ 21.6·e^(0.94h)).
   - A persistent q = 0.01 alarms in 4 ticks at 900 s and in 6 ticks at 60 s.
   - A daily block-bootstrap audit on one entity per hour checks the realised rate and raises h by at most 20% if it exceeds 2× the target.
4) Paths: single tick when e_day ≤ 0.03·alpha_mult; evidence alarm; acc_alarm from any accumulator.
5) Severity from e_day: ≤ 0.03 LOW, ≤ 3e-3 MEDIUM, ≤ 3e-4 HIGH candidate, ≤ 3e-6 CRITICAL candidate.
   - HIGH needs at least 2 axes with family e_day ≤ 0.03 within 4 ticks, OR 2 consecutive ticks ≤ 3e-3, OR a discrete finding ≥ HIGH.
   - CRITICAL needs 3 or more axes or lib-4 ≥ high.
   - An evidence-only alarm is LOW (MEDIUM if S ≥ 2h).
   - An accumulator alarm uses the e_day of its own p.
6) Reading rules, by axis:
   - volume-only is capped at MEDIUM for an entity and LOW for a class;
   - self/peer 2×2: self abnormal with peer p > 0.1 lowers one level; both abnormal raises one level;
   - the common-mode flag lowers volume, transport and app-error evidence one level only;
   - schedule_shift-explained temporal evidence is capped at LOW;
   - class keys: a coherent intensity-only change or a system-wide app-error/transport change is capped at LOW.
7) Across a system's entities per tick, apply BH at q = 0.05 to the LOW single-tick path only.
8) Write behavior.alarm {path, severity, axes, families}.

**Reads.** behavior.p, behavior.axes, behavior.acc_alarm, model.feedback, behavior.calib_health, behavior.common.flag, store.events(since=t−Δt, kinds=discrete), store.matches(since=t−Δt), feature.tctx

**Writes.** behavior.p_family, behavior.q_inst, behavior.q_all, behavior.e_day, behavior.evidence, behavior.alarm, model.calib meta rings

**Perf.** ≈1 ms per tick

**Unit test.** (a) Masking regression: [1e-3, 0.999, 0.5] gives p_family ≈ 3e-3.
(b) Five nulls with ρ = 0.64 over 2e5 ticks: the rate at 1e-3 is within [0.8, 1.3]× before meta-calibration and within [0.8, 1.2]× after.
(c) A uniform null at Δt = 900 and at Δt = 60 for 200 simulated entity-days: evidence alarms per entity-day within [0.015, 0.045] at both.
(d) A persistent q = 0.01 alarms by tick 4 at 900 s and tick 6 at 60 s; an isolated q = 0.005 does not alarm.
(e) Volume-only at e_day 1e-6 gives MEDIUM. Adding categorical at e_day 0.01 allows HIGH.
(f) One family at e_day 1e-5 for a single tick with no corroboration gives MEDIUM.

**Integration notes (as built, docs/lib3/integration.md).**
- Unit test (c): the [0.015, 0.045] band is checked over replicates (a single 200 entity-day sample holds ~6 expected alarms).
- The evidence CUSUM restarts at 0 on the first live tick after warm-up.

**Canonical grain mode (cadence.md §7.3–§7.5, §9.2, §17).**
- Tick type τ ∈ {h, q, t}; meta rings keyed by (daypart, τ[, cc]); single-tick `e_day = q_all · n_τ / β_τ` with β = (0.5, 0.25, 0.25) renormalised over the types present (0.03 null alarms per entity-day in total at every Δt; identical to `q·86400/Δt` at 3600 s).
- Two evidence CUSUMs, each with half the 33-day budget (ARL 66 d): S_t (`behavior.evidence`) over the T-stream instantaneous detectors every tick, h = 6.05 at 900 s and 8.93 at 60 s; S_h (`behavior.evidence.h`, input `behavior.q_inst.h`) over the H-stream instantaneous detectors except the overlapping identity, once per hour, h = 4.57 at every cadence. The Q stream feeds no CUSUM. The per-stream audit is implemented. The S_h reset at the first live tick persists even when that tick is not an H tick (cadence.md §17).
- Corroboration windows are wall-clock per stream (max(4Δt, 1 h) for H evidence, max(4Δt, 15 min) for Q); provisional Q evidence is weighted × 0.5, can raise at most MEDIUM and never counts toward HIGH / CRITICAL corroboration. The pending entry records (stratum, τ, Δt) for B29.
- Measured (integration.md §10.5): replaying other β shares on the recorded per-tick e_day moves the null single-tick rate by ±15–25 % only; the realised evidence-CUSUM rate is ~25× its budget whatever the split.

## B26 — RiskEngine [new]

- **File:** `backend/app/engines/behavior/risk.py`
- **Layer / order / interval:** behavior / 26 / 1

**Purpose.** (Previously B24.) One explainable, decaying 0–100 risk per entity and per class. It accumulates weak heterogeneous evidence independently of cadence, saturates within an episode, maps evidence to kill-chain stages, and cannot be poisoned (fixed L_ref).

**Algorithm.**

Evidence per tick:
- Continuous: b_f = W_f·max(0, log10(1/e_day(p_family))), i.e. excess surprise over what is expected once per day, so the null input per day does not depend on Δt. W: intensity 1, shape 1.5, categorical 1.5, identity 2, c2 2, exfil 2, breadth 1.5, change 1.5, temporal 1, sequence 1.5, peer 1.
- Episode saturation: within one open incident or episode, the n-th consecutive tick's contribution from the same (entity, family) is multiplied by 0.5^((n−1)/4), so the total is ≤ 6.3·b.
- Discrete events: first_seen system 15, class 8, entity 3; rare_access 15; client_impersonation 20; identity_mismatch 20; possible_impersonation 20; new_entity_unmatched 20; beacon 20; budget_exceeded 20; class_adoption_risky 20. The n-th repeat of a key within 24 h counts ×0.5^(n−1).
- lib-4 matches (one-tick lag): info 0, low 5, medium 15, high 30, critical 50, × confidence.
- Suppressed incidents count ×0.25.
Decay: L_k ← L_k·2^(−Δt/H_k), with wall-clock H = 12 h for behaviour, 24 h for change and sequence, 72 h for novelty and identity, 48 h for c2, exfil and critical.
Stages (lib/stages.py): M = min(2.2, 1 + 0.3·(#stages in 24 h − 1)).
Risk = 100·(1 − exp(−M·c_s·π·ΣL/L_ref)), where c_s ∈ [0.5, 2] is criticality, π is the feedback multiplier and L_ref = 60 is fixed. L_ref is tuned offline on clean eval replays and never set from live data.
- Simulated null (8 families): mean L ≈ 7 and p99.99 ≈ 17–19 at both 60 s and 900 s → risk ≈ 11 at the mean and ≤ 27 at p99.99.
- Common-mode-flagged volume evidence is multiplied by 0.3.
Tiers: Low < 30 ≤ Medium < 60 ≤ High < 85 ≤ Critical, with −10 hysteresis.
Class risk = max(the class pseudo-entity's own risk from its own detectors, mean of the top 3 members). System risk = max over entities and classes.

**Reads.** behavior.p_family, behavior.alarm, store.events(since, kinds), store.matches(since), store.incidents(status=open), model.feedback, behavior.common.flag, ctx.config.criticality

**Writes.** behavior.risk@(s, e|class:<id>|__system__), profile.extra.risk

**Perf.** Less than 1 ms per tick

**Unit test.** (a) Three families at e_day 0.01 each for 12 ticks at 900 s: L ≈ 49 and risk ≥ 30 (Medium). Adding a class-tier first_seen (a second stage) gives risk ≥ 60 (≈71).
(b) A null over 60 simulated days at Δt = 900 and at Δt = 60: risk < 30 on ≥ 99.99% of ticks, and mean L within [5, 10] at both.
(c) 36 repeats of one event key contribute ≤ 2× a single one.
(d) Risk halves after H with no new evidence.

**Integration notes (as built, docs/lib3/integration.md).**
- Warm-up evidence does not carry into live risk: the per-key state is cleared on the first live tick after training (the warm-up risk series stays in the store).
- Habitual lib-4 activity: a match of severity ≤ medium whose (entity, signature) has matched on ≥ 4 ticks, the first ≥ 24 h earlier, weighs 0. lib-4 severities grade activities (routine login / form write / admin page / poor TCP are 'low'), so without this every busy entity carried L ≈ 30–50 of routine matches (risk 40–60, plus credential / privilege / exfiltration stages from the auth / admin / transfer categories) on every live tick, far above the spec's null (L ≈ 7). A new routine activity counts for its first day; high and critical matches always count; habits are learnt in warm-up too and forgotten after 30 d unseen.

**Canonical grain mode.** Continuous evidence uses `excess_surprise(p, period_s)` with the family's stream period (cadence.md §9.3), so an hourly family earns the same per-day evidence at every cadence.

## B27 — IncidentEngine [new]

- **File:** `backend/app/engines/behavior/incident.py`
- **Layer / order / interval:** behavior / 27 / 1

**Purpose.** (Previously B25.) The only notifier. Turns alarms and discrete findings into a few entity-centric and class-centric incidents, with a lifecycle whose closing does not depend on risk. Also handles common-mode parenting, campaign merge, suppression and a notification budget.

**Algorithm.**

Opening (suppressed during ctx.training), on any of:
- an alarm (single-tick, evidence or accumulator);
- a discrete finding ≥ MEDIUM;
- a class alarm at a class key;
- risk ≥ Medium for 2 or more ticks together with at least one family at e_day ≤ 0.1 in the last 24 h, the family hit newer than the key's last incident activity and the risk ≥ Medium beyond what that incident already covered (eval round 3, see the integration notes).
Join: the open incident of the same entity, continuity alias or actor, if the gap is ≤ max(4 ticks, 1 h).
Escalate: on a severity increase or a new axis.
Close, when any of these holds (never based on risk):
- (a) the governor regime becomes RETURNED;
- (b) the governor ACCEPTs, close_reason = accepted;
- (c) a label closes it;
- (d) no alarm or finding for max(8 ticks, 2 h), every accumulator < h/4, and e_day(q_inst) ≥ 1 for the last 4 ticks.
An incident held longer than 14 d without resolution is sent to the label queue, not auto-closed. Reopening within 24 h reuses the id.
Common mode: when at least 50% of a class alarms in the same tick with axes only in {volume, transport, app-error}, the member incidents become children (status suppressed_common) of the class incident from B18. This does not apply to members that have other axes.
Campaign: open incidents in the same system within 1 h whose pattern tokens (axes, feature signs, new tokens) have Jaccard ≥ 0.5 are merged by union-find under a campaign_id.
Feedback policies are applied.
Token buckets: 3 per entity per hour and 20 per system per day; overflow is queued by risk.
Emit BehaviorEvent(kind='incident', extra.state, e_day, axes, p_by_detector, incident_id), and store the Incident.

**Reads.** behavior.alarm, behavior.p_family, behavior.e_day, behavior.risk, behavior.acc_alarm, behavior.regime, store.events(since, kinds), store.matches(since), model.feedback, model.link, model.class

**Writes.** store incidents (entity and class); BehaviorEvent kind='incident'; event status and incident_id updates

**Perf.** Less than 1 ms per tick

**Unit test.** (a) 40 consecutive alarm ticks on one entity give exactly 1 incident and at most 3 notifications.
(b) 5 of 8 class members alarm on the volume axis only: 1 class coherent_shift parent and 0 notifying entity incidents.
(c) 3 entities with the same new SNI token within 1 h: 1 campaign.
(d) After an attack ends with risk still ≥ 60, the incident closes within max(8 ticks, 2 h) when the accumulators have decayed.

**Integration notes (as built, docs/lib3/integration.md).**
- Every accumulator, cusum / mcusum included, is tested on the calibrated-p level lib/detectors.acc_level (shared with B28) in the quiet close; m_cp.level is diagnostic only.
- A HABITUAL lib-4 match (lib/stages.habit_step, the rule B26 uses: severity ≤ medium, ≥ 4 matched ticks of the (entity, signature), the first ≥ 24 h earlier; learnt from every tick, warm-up included) joins as evidence but does not restart the quiet clock (eval round 2): an integration host's routine 'high_error_backend' (medium) match on almost every tick kept an FP incident open for days, and the T4 exfiltration that started meanwhile only escalated it. Lib-4 evidence entries: ≥ MEDIUM at most hourly per signature, below MEDIUM once per signature per (re)opening (a poller's routine info matches filled 477 of the 512 kept evidence entries and evicted the alarm entries).
- Risk opening needs NEW risk (eval round 3, integration.md §10.3): B26's risk is 100 (1 − e^−x) with x additive in the evidence, and after a quiet close it decays over days (half-lives 12–72 h), while a family at e_day ≤ 0.1 is an ordinary null event (~0.1 per family and entity-day). With only the family hit required to be new, the old risk plus any later weak hit reopened the incident within hours (pack A seed 0: 58 reopenings of 23 control incidents, median 6 h after the close; pack B: the sanctioned health checker 10.40.9.9 reopened ~40 times on risk alone). The risk trigger now also requires 100 (1 − e^−(x_now − x_old)) ≥ 30, where x_old is the key's risk at its last incident activity decayed with the slowest B26 half-life (72 h, an upper bound of what the covered evidence still contributes). Alarms and findings still reopen as before. Test: test_b27_incident.py::test_decaying_risk_of_a_closed_incident_does_not_reopen_it_on_a_weak_hit.

**Canonical grain mode.** `acc_level_from_p(p, d, period_s(d, Δt))` counts ARLs in grain periods for H accumulators; the quiet close needs both q_inst,t and the latest q_inst,h (≤ 1 h old) to be quiet (cadence.md §9.3).

**Metric note (integration.md §10.7 item 5).** An attack on an entity whose LOW FP incident is already open joins and escalates that incident (CRITICAL, expected axes, an 'escalate' notification); eval.md counts only openings in the scenario window, so these are scored as misses. The risk-opening fix above cuts openings by 10–48 % but raises distinct incident ids (FAR per id).

## B28 — GovernorEngine [new]

- **File:** `backend/app/engines/behavior/governor.py`
- **Layer / order / interval:** behavior / 28 / 1

**Purpose.** (Previously B26.) Model governance that resists poisoning, for entities and for classes. Covers trust and provisional trust, quarantine, a regime state machine with legitimate-versus-attack reasoning (including time-in-regime evidence), retroactive rollback, champion/challenger versioning, a cold-start rule that cannot deadlock, and no permanent lockout.

**Algorithm.**

Trust (architecture §3):
- trust_prov = clip(log10(e_inst/0.1), 0, 1) × [no alarm] × [no discrete finding ≥ MEDIUM];
- trust = trust_prov × [no open incident] × [regime NORMAL or RETURNED] × [every accumulator < h/2];
- ctx.training ⇒ 1, unless lib-4 ≥ HIGH;
- quarantine = open incident OR regime ∈ {SUSPECT, DRIFTING}.
Regime machine:
- NORMAL → SUSPECT when any accumulator ≥ h/2, cp.prob > 0.7, baseline_creep or an alarm occurs. The onset τ̂ comes from behavior.cp.onset, else t − Δt.
- If τ̂ is before the commit frontier t − D·Δt, write model.control.rollback_to = τ̂ − Δt (at most once per hour, at most 7 d back) and emit regime(state = rollback).
- SUSPECT → DRIFTING after ≥ 1 h and ≥ 4 ticks.
- → RETURNED when every accumulator < h/4 and there is no alarm for max(8 ticks, 2 h); write release = [τ̂, t].
- → ACCEPTED or REJECTED as below.
Type is taken from the evidence axes: intensity, shape, categorical, rhythm, ramp, identity or new_entity.
Legit log-odds:
- Prior by type: intensity 0, rhythm 0, ramp +0.5 (only if |Sen slope| ≤ 0.05 log-units per day and residuals around the trend are stationary), shape −1, categorical −1.5, new_entity −1, identity −3, c2/exfil −4.
- Add ln LR terms:
  - peer concordance (≥ 50% of the class moving in the same direction within ±1 h): +2.08;
  - lib-4 ≥ high: −2.3;
  - system-tier novelty with IDF > ln(N/2), or an external upload destination: −1.6;
  - identity self-posterior ≥ 0.9: +0.69; identity mismatch or client concurrency: −2.3;
  - post-change dispersion ratio ≤ 1.5: +0.69;
  - time in regime: +ln 2 per T_type/3 of stationary, clean duration (Mann–Kendall p > 0.1 on the within-regime residuals, and no new axes), capped at +ln 8. T_type is 1 d for intensity, ramp and new_entity; 3 d for rhythm; 7 d for shape and categorical; identity never accrues time evidence.
  - expected_change label: accept; malicious label: reject and freeze.
- Arithmetic:
  - single-entity intensity: 0 + 0.69 + 0.69 + 2.08 = 3.46, P = 0.97 → accepted at 1 d;
  - shape: 2.46, P = 0.92 → accepted at 7 d;
  - categorical: 1.96, P = 0.88 → needs peer concordance or a label.
ACCEPT when P ≥ 0.9, duration ≥ T_type, and there is no malicious-type evidence (lib-4 ≥ high, identity mismatch or impersonation, beacon alarm, exfil-axis budget alarm, system-tier sensitive novelty). Risk tier is not a condition.
- Write model.control {version + 1, rebase_from = τ̂, allow_drift = the slope for a ramp}; learners replay the held rows.
- Emit regime(state = accepted) at INFO and close the incident.
REJECT when P ≤ 0.2 and the episode is corroborated (W7 tuning, m_governor.reject_corroborated: its own alarm / accumulator evidence on ≥ 4 ticks spanning ≥ 1 h — the SUSPECT → DRIFTING persistence — or two independent malicious sources, or a tp label): stay quarantined and discard the held rows. One lib-4 ≥ HIGH match meeting a one-tick rhythm alarm froze entities on their first live tick.
trust_evidence (W7 tuning), live ticks only = the live trust's gate on the row itself ([no alarm] × [no finding ≥ MEDIUM] × [every accumulator < h/2]); B24 and B25 cap their ring admission with it, which binds for released rows (trust_prov has no accumulator factor). It carries neither the regime state nor the q_inst evidence factor (B25's own output), and is not written in training: gating warm-up rows on their own evidence truncated the meta rings' null (pack A: single-tick exceedance 5× → 18× nominal, 62× with the q factor).
Otherwise hold as DRIFTING at LOW, update every 24 ticks, and push to the label queue after 14 d.
A class-wide change accepts at class level after 24 h of concordance. Profile versions are kept via put_profile_version.

**Reads.** behavior.alarm, behavior.q_inst, behavior.e_day, behavior.acc_alarm, behavior.score (accumulators), behavior.cp.*, behavior.axes, behavior.id, store.incidents, store.events(since, kinds), store.matches(since), model.feedback, model.class, model.baseline (slope diagnostics)

**Writes.** behavior.trust, behavior.trust_prov, behavior.quarantine, behavior.trust_evidence, behavior.regime, model.control@(s,e|class:<id>), model.governor, profile.extra.regime, profile versions; event regime

**Perf.** ≈1 ms per tick; rollback replays are O(rows × F) and occur at most hourly

**Unit test.** (a) An exfil alarm for 160 ticks: trust = 0 throughout, no version bump, REJECTED once a lib-4 high match appears.
(b) 6 of 8 class members shift together for 2 d: accepted, with rebase_from ≈ onset ± 4 ticks.
(c) A single-entity doubling with no corroboration: DRIFTING at LOW, then accepted at 1 d.
(d) A slow ramp detected at t with τ̂ = t − 40 ticks: control.rollback_to = τ̂ − Δt, and the model.baseline statistics equal a fit on rows ≤ τ̂.
(e) Training on an empty store: trust = 1 on every warm-up tick, and on the first live tick a normal entity has trust > 0 (deadlock regression).
(f) The incident closes when RETURNED even though risk ≥ 30.
(g) A +15%/day ramp is never accepted; a +2%/day ramp is accepted after 1 d of stationarity.

**Integration notes (as built, docs/lib3/integration.md).**
- SUSPECT on an accumulator needs level ≥ h/2 AND e_day(p) · n_scored ≤ 0.1 (or its acc_alarm); REJECT needs evidence beyond the type prior; ACCEPT durations and the 14-day label queue are time in regime; the release starts at min(τ̂, q_floor); test (d) fits on rows ≤ τ̂ − Δt.
- Regime types include c2 and exfil (contract F).
- A rollback never reaches into warm-up rows (trusted by definition): rollback_to is floored at the last training tick. Without this floor an onset estimated at the start of the data erased the whole model and cascaded into REJECT / freeze of clean entities.
- Link retractions (m_link.retractions) are answered with model.control {rollback_to: t_link, release: [t_link, now]} on the seeded entity, once per link, deferred while a rollback is rate-limited.
- Label-queue keys (DRIFTING > 14 d, m_governor.label_queue) are queued by B23 as reason 'held', source 'governor'.

**Canonical grain mode.** `evidence_factor` takes e_inst = min over streams of q_inst,s·N_s (N_t = 86400/Δt, N_h = 24); `_ln_arl` works per detector period; the level and ramp series are H zr at H ticks (held between them) with hourly Sen-slope bins (cadence.md §9.3).

**Open (integration.md §10.7 item 1).** Training trust is 0 on warm-up ticks with a lib-4 match ≥ HIGH, so a sanctioned nightly backup that matches `bulk_upload` (HIGH) every night never gets a trusted night hour; B13 then falls back to the peers' level and L15 is CRITICAL in every run. Needs a design decision (habituation of recurring same-phase HIGH matches, a sanctioned-automation allowlist, or keeping training trust).

## B29 — ExplainEngine [new]

- **File:** `backend/app/engines/behavior/explain.py`
- **Layer / order / interval:** behavior / 29 / 1

**Purpose.** (Previously B27.) A faithful, testable explanation for every incident that opens or escalates. It gives attribution in natural units against the normal range for this hour and day type, what is new, a minimal counterfactual recomputed exactly (stateful detectors included) by deterministic replay, peer context, and a narrative.

**Algorithm.**

Runs only on incidents that opened or escalated this tick.
1) Numeric attribution:
   - predictive intervals from profile.extra.model_state, e.g. 'bytes_up 38.2 MB/15 min; usual Tue 14:00 0.4–1.1 MB (35×)';
   - RBC χ²₁ p-values and Garthwaite–Koch shares c_i = w_i²/D² with w = Σ^(−1/2)·z;
   - BH at q = 0.05.
2) Categorical: new tokens with tier and first_seen, vanished habitual tokens (share ≥ 2% for ≥ 3 days), client stack diff.
3) Sequence: the 3 least likely transitions.
4) Faithful counterfactual via lib/replay.py.
   - Greedily reset the top attributed features to their bucket median (numeric) or remove the new tokens (categorical).
   - Recompute every affected detector exactly:
     - stateless detectors through their pure helpers against the stored models;
     - stateful detectors (CUSUM bank, MCUSUM, rhythm W, budget sums, identity CUSUMs) by replaying the stored input history (behavior.zr, behavior.cusum_state, act.* rows) over the accumulator window from τ̂;
     - then p-values via calib.p_from_ring on the model.calib snapshot, wHMP, meta-calibration and the decision rule.
   - Stop when the incident's opening condition no longer holds. Report the minimal set, validity (did the decision flip), and scope (which detectors were recomputed).
5) Peer context from behavior.common and the class members' current p.
6) Nearest labelled pattern by cosine of the deviation vector.
7) Deterministic zh/en templates: a headline and 3 bullets.
The z shown in the UI is the engine's own z, fixing routes.py:140.

**Reads.** store.incidents, profile.extra.model_state/last_contrib/categorical/client_stacks/sequence, behavior.pf, behavior.z, behavior.zr, behavior.cusum_state, model.density, model.calib, all detector models via get_model, store.events(since), behavior.common.*, model.feedback

**Writes.** incident.explanation {attributions, new_tokens, vanished, counterfactual_set, counterfactual_valid, counterfactual_scope, nearest_pattern, peer_context}, incident.narrative_zh/en

**Perf.** ≤10 ms per incident (replay ≤ 96 ticks × 48 charts)

**Unit test.** Build an incident where bytes_up and updown_log are perturbed by +4σ for 12 ticks (so the CUSUM alarms) and a new SNI appears.
- The top-3 attributions include {bytes_up, updown_log}.
- new_tokens contains the SNI at class tier.
- The counterfactual set is a subset of the perturbed features and the counterfactual is valid: the replayed CUSUM stays below h and the fused decision is recomputed as no incident.
- The narrative contains the natural-unit range.

**Canonical grain mode (cadence.md §9.4, §17).** Numeric attribution uses the grain whose detector holds the smallest p (Q when a `_q` detector drives), with natural units per hour or per 15 min from the per-grain model_state; neutralising resets the grain value to that grain's bucket median. The replay recomputes the H / Q rows scored at the tick, the CUSUM bank over ≤ 96 H states and each evidence CUSUM stream over its own excursion, p through `p_replay(grain, prov)` (pm rings included), the meta strata B25 recorded, and e_day = q_all·n_τ/β_τ. `counterfactual_scope` lists the grains recomputed. Latest gate 10: hit@3 0.25, counterfactual validity 0.35 (targets 0.8 / 0.9).

## B30 — PortraitEngine [new]

- **File:** `backend/app/engines/behavior/portrait.py`
- **Layer / order / interval:** behavior / 30 / 1 (each key refreshes once per 2 h at its own phase)

**Purpose.** (Previously B28.) The human-readable, dynamically generated behaviour library: versioned portraits with diffs, for each IP and for each class (role, sub-class, static CIDR and pool).

**Algorithm.**

Runs every 8 ticks or 2 h and assembles:
- Identity: IP or class, system, static class, class_path with probability, identifiability with a Wilson CI, confusable_with, continuity and aliases, shared_ip.
- Role: automation index; activity mix from action-token families and lib-4 categories over 7 d.
- Rhythm: the 80% window, weekday/weekend ratio, periodicity, schedule-shift history.
- Workload: p10/p50/p90 per day type in natural units from model.baseline.
- Client stacks; top SNI, ports and peers; top templates and top-3 n-grams; timing (burstiness, think time).
- Distinctive traits.
- Stability: version, branch, regime state and history (including rollbacks), maturity = 1 − Good–Turing unseen mass per dimension, n_eff per bucket.
- Risk and its top reasons; the last 5 timeline items (store.timeline, indexed).
Class portraits additionally include:
- members with their probabilities, cohesion, common tokens (df ≥ 50%), class-versus-system distinctive tokens, outlier members;
- the class_monitor aggregate bands and active-fraction heatmap, the adoption history, and class risk.
Diff against the previous version: categorical JSD > 0.1, quantile shift > 25%, window change > 1 h.
Rendered with zh/en templates, e.g. '10.20.1.12：erp-prod 人工交互用户(同类4个,置信0.92)，工作日08:30–18:40活跃，典型每刻6–14请求…可辨识度0.94(易混淆:10.20.1.11)'.
Personal data from paths and query values is never embedded; only templates are used.

**Reads.** profile.extra.*, model.baseline, model.classagg, model.vocab, model.rhythm, model.client, model.seq, model.timing, model.class, model.identity, store.timeline

**Writes.** profile.extra.portrait {json, text_zh, text_en, version, diff} for entities and classes; put_profile_version

**Perf.** ≈1 ms per entity or class per run; ≤5 ms per tick amortised

**Unit test.** Build synthetic models: rhythm active 09–18, top path /orders at 40%, stack chrome126.
- The portrait window is '09:00–18:00' ± 30 min and the top template is /orders.
- After the window changes to 14–23, the diff reports that 活跃时段 changed.
- A static CIDR class of 3 IPs gets a portrait with members and class bands.

**Integration notes (as built, docs/lib3/integration.md).**
- The engine runs every tick and each IP / class key refreshes once per refresh_s = 2 h at its own crc32 phase (Engine.entity_due). An 8-tick engine stride refreshed every key on the same tick (≈ 70 ms bursts at 20 entities); the per-key phase spreads the same work.

**Canonical grain mode (cadence.md §9.5, §17).** Workload bands p5/p50/p95 per grain and day type under `workload.<f>.grains.{h,q}`: the H band is the H anchor's own predictive per hour, the Q band its transfer per 15 min with a `provisional` flag; the legacy per-tick band is the Q band (exposure 900 s). Class bands come from B18's hourly aggregate anchors. The zh / en texts name the grain. Public helpers used by the API live in `lib/m_portrait` (signature, diff_signatures, safe_token). Latest gate 11: H-row coverage 0.93 and Q-row 0.95 (in [0.85, 0.95]); per-tick coverage 0.96 (just above the band).
