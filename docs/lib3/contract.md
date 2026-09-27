A. FEATURE_SPEC v2 (engines/behavior/lib/features.py)

Each entry is (name, source, kind, n_source, group). The module exports FEATURE_NAMES_V2, FEATURE_KIND, FEATURE_GROUP, FEATURE_NSRC, KEY_FEATURES and FEATURE_DIM (= 52).

Kinds and transforms:
- count / bytes: vec = log1p(v·60/Δt). A stale source gives a true 0.
- ratio: k/n with n taken from n_source. vec = logit((k+0.5)/(n+1)); nat = k/n. A stale source or n = 0 gives NaN.
- avg: mean over n items. vec = log(v); nat = v. A stale source or n = 0 gives NaN.
- bounded: a [0,1] descriptor. vec = logit(clip(v, 1e-3, 1−1e-3)). Requires n ≥ 5, otherwise NaN.
- gauge: fresh values only.
- window: derived window descriptor. Valid when it was written this tick.
- clr: CLR of (x + 0.5).
- ctx: context only, not scored.

Features by group:
- volume: bytes_up(l4.bytes_up, bytes), bytes_down(l4.bytes_down, bytes), flows(l4.flows, count), http_requests(http.requests, count), dns_queries(dns.queries, count), tls_handshakes(tls.handshakes, count), intensity(act.events, count), bytes_per_flow(l3.bytes_total/l4.flows, avg, n = l4.flows), updown_log(log((l4.bytes_up+1)/(l4.bytes_down+1)), avg, n = l4.flows).
- breadth: distinct_peers(l4.distinct_peers, count), distinct_dports(l4.distinct_dports, count), distinct_templates(act.distinct_templates, count), new_peer_count(derived.new_peer_count, count), dest_concentration(derived.dest_concentration, bounded, n = tls.handshakes + dns.queries).
- app: http_write_ratio(http.write_count/http.requests, ratio), http_get_ratio(ratio), http_4xx_rate(http.status_4xx/http.requests, ratio), http_5xx_rate(ratio), http_3xx_rate(ratio), http_latency(http.latency_ms_avg, avg, n = requests), resp_bytes_avg(avg), req_bytes_avg(avg), path_entropy(derived.path_entropy, bounded, n = derived.path_entropy_n), new_template_ratio(act.new_template_ratio, ratio, n = http.requests).
- dns: dns_name_entropy(bounded), dns_dga_score(avg, n = dns.queries), dns_fail_rate(ratio), dns_txt_ratio(dns.txt_count/dns.queries, ratio), dns_qname_len(avg).
- tls: sni_entropy(bounded), tls_weak_ratio(ratio, n = handshakes), ja3_diversity(derived.ja3_diversity, count), tls_handshake_ms(avg).
- timing: periodicity(window), timing_regularity(window), think_time(derived.think_time_s_avg, avg, fresh only), req_per_session(window), duty_cycle(window).
- transport: retransmit_rate(ratio, n = pkts), rtt(avg, n = flows), syn_ratio(l4.syn_count/l4.flows, ratio), flow_duration(avg).
- probe: probe_reachable(gauge), probe_loss(ratio, n = probes).
- comp: comp_get, comp_write, comp_4xx, comp_5xx, comp_dns, comp_tls, comp_flows, comp_syn (clr).

Removed from v1: fanout (it duplicates distinct_peers, graph.py:45-46) and peer_novelty (moved to B08; derived.peer_novelty is kept for lib-4).

KEY_FEATURES (12, used by the CUSUM bank): bytes_up, bytes_down, flows, http_requests, dns_queries, tls_handshakes, distinct_peers, distinct_templates, http_write_ratio, http_4xx_rate, updown_log, bytes_per_flow.

EXPOSURE_CHANNELS = [http, dns, tls, flows, probe].

B. Series

Keys have the form '<system>|<entity>|<name>'.

Pseudo-entities are '__system__', ('__org__', '__org__'), 'class:<role_id>', 'class:static:<name>' and 'class:pool:<cidr>'.
- They are written only through add_vec, add_derived, put_model and put_profile. add_raw rejects them.
- store.entities() excludes them; store.pseudo_entities(system) lists them.
- A class key always lives under the real system it describes.

Raw (lib-1):
- Existing names, with full sets of up to 64 entries plus '__other__'.
- New: l4.dport_set, l4.peer_set, l4.syn_count, l4.pkts_total, l4.flow_duration_ms_avg, http.status_3xx, http.get_count, http.write_count, http.req_bytes_avg, dns.txt_count, dns.qname_len_avg, tls.handshake_ms_avg.
- act.tokens {token: n}.
- act.stream: numpy structured rows [ts, token_id, outcome, up, down, dest_id, stack_id], at most 512 per tick.
- act.stream_frac, act.events (zero-filled with touch=False), act.objs {template: {n, ids ≤ 256, hll?}}, act.distinct_templates, act.new_template_ratio (absent until the system vocabulary model.template is 24 h old: before one daily cycle, "new to the system" measures the vocabulary's own growth, 1.0 on the first tick).
- act.rare_events {dest_id: [[ts, up, down]]}.
- client.stack_set {token: {n, bytes, first_ts, last_ts}}, client.stack_events [[stack_id, first_ts, last_ts, n]], client.ua_set, client.ja3n_set, client.ttl_set, client.os_ua_ttl_pairs.

Derived (lib-2):
- Existing names.
- Instant metrics are emitted only when fresh, plus derived.<ratio>.n and derived.<entropy>_n.
  - derived.<entropy>_n is the total mass of the source set INCLUDING '__other__' (the true request / query / handshake count, R1's full sets); the entropy itself is computed over the named entries only ('__other__' is a remainder, not a category). B01's bounded gate (n ≥ 5) and B04 read this n.
  - derived.<ratio>.n is the ratio's trial count (dns_fail_rate: dns.queries).
  - Graph peers (new_peer_count, peer_novelty, dest_concentration) are TLS / DNS names at eTLD+1 plus l4.peer_set buckets (/24, /64 or host); '__other__' is never a peer.
- Window metrics carry dims {span_s, n_active}.

Behavior (lib-3). All vector series are float32 rings created with store.add_vec.
- Features: feature.vec[52], feature.nat[52], feature.expo{channel: n}, feature.active, feature.tctx{hour_local, dow, day_type, bin48, bin168, slot, daypart, cc}, feature.sketch[80].
- feature.<name> scalars are VIRTUAL column views of feature.vec, never stored twice.
- Per-feature: behavior.z[52] (against the current anchor), behavior.zr[52] (against the reference anchor), behavior.pf[52], behavior.zi[52], behavior.wh.
- Detector outputs:
  - behavior.score[D], behavior.pm[D] and behavior.p[D] are aligned to DETECTORS (D = 31). NaN means unscored or degraded.
  - behavior.score.<d> and behavior.p.<d> are virtual views.
  - behavior.axes {d: [axis]}, behavior.acc_alarm {d: 0|1}, behavior.degraded {d: cause}.
- Fusion: behavior.p_family {family: p}, behavior.q_inst, behavior.q_all, behavior.e_day, behavior.evidence, behavior.alarm {path, severity, axes}.
- Governance: behavior.trust, behavior.trust_prov, behavior.quarantine, behavior.regime.
- Risk: behavior.risk at the entity, at class:<id> and at __system__.
- Changepoint: behavior.cp.prob, behavior.cp.onset, behavior.cusum_state (a vector used for replay).
- behavior.rhythm {p_expected, W_off, s_sil}.
- behavior.timing {B, M, think_mu, think_sigma, period, period_p}.
- behavior.budget {'<Q>.<H>': [value, z_q, tail_p]}. This single dict series replaces behavior.budget.<Q>.<H>.
- behavior.id {posterior_self, best_other{}, p_unknown, cusum_other, cusum_new}.
- behavior.seq.class_llr.
- behavior.embed.session[16] (P2).
- behavior.common.<group> at '__system__' and at 'class:<id>' (role, static and pool classes), group ∈ {volume, transport, app_error, probe}: dict {L (leave-one-out common loading, median), n (pool size), n_scored, n_members, frac_up, frac_down, dir ∈ {-1, 0, 1}, dt, run (consecutive ticks in the same direction)}. B18 and B25 read it at ts = now (B05 runs before them).
- behavior.common.flag at each real entity, {group: 0|1}.
- behavior.class {active_frac, coherence, ...} at 'class:<id>' (dict) and behavior.class.agg[52] (float32 vec ring, B18's aggregate row and its reference learner's replay clock), plus behavior.class.new_ext.
- behavior.calib_health at '__system__', {d: {ks, rate_ratio, weight_mult}}.
- ops.engine_health at '__system__'.

Retention (store.set_retention(prefix, max_points, max_age_s); pruning on append):

| Series | max_age |
|---|---|
| raw scalars | 6 h |
| D0 grid inputs http.requests, l4.flows, dns.queries, act.events (raw) | 24 h (D0 and D2 spans) |
| D0 trend targets l4.bytes_up, l4.distinct_peers, http.latency_ms_avg, probe.rtt_ms (raw) | 13 h (12 h trend span + 1 h) |
| raw categorical sets, act.stream, act.rare_events, client.* | 1 h |
| derived.* | 2 h |
| feature.nat, expo, active, tctx; behavior.trust, trust_prov, quarantine, regime, risk, q_inst, q_all, e_day, evidence, alarm; behavior.calib_health | 8 d (needed for rollback and rebase replay) |
| behavior.class.agg | 9 d (B18's reference commits 24 h late and replays 8 d) |
| behavior.budget | 1 d and at most 24 points (36 entries per point) |
| feature.vec, feature.sketch, behavior.score, pm, p, p_family, axes | 1 d |
| behavior.z, zr, zi, pf, cusum_state | 6 h |
| events | 30 d |
| incidents | 90 d |
| labels | never pruned |
| profile versions | last 12 |

These rules are the store's DEFAULT_RETENTION (core/store.py); engines no longer need to set them.

Scalar rings. behavior.risk (entity, class:<id>, __system__) is a 1-element float32 vec ring like trust / q_all / e_day (helpers_api §0.1); read it with vec_at / vec_tail / vec_since (store.timeline lists a risk point when it enters a new 10-point band). profile.extra.risk.tier names are lowercase: 'low', 'medium', 'high', 'critical'.

C. Models (put_model(system, entity, name, obj, version) / get_model / model_version)

Each model has exactly one owner engine.
- model.template@(s, __system__): R2.
- model.baseline@(s, e): B03. Fields {fmt, tier='entity', current: Anchor, reference: Anchor, gate, gate_ref (GateState), golden {stats, snaps}, ref_elig, n_eff, version, branch, held [CommitRow], allow_drift}. Anchor stats are stored per family (k = 6 count, 4 ratio, 3 t; see lib/m_baseline); consumers use the m_baseline accessors. There is no slope_log field: B28 takes a ramp slope from B14's baseline_creep event, else its own Sen slope.
- model.baseline@(s, class:<rid>) holds the pooled member predictive; model.baseline also exists at @(s, __system__) and @(__org__, __org__). Tier models: {fmt, tier ∈ {class, system, org}, ts, version, stats[48, L], E, h, kappa, kappa_cls, members, n_eff}, recomputed hourly from the members' current anchors.
- model.class@(__org__, __org__): B02. Fields {assign{'sys|ip': {role, sub, prob, static[], pool, super, class_path, A, provisional?, pend?, D?}}, roles{rid: {name, members, medoid, lineage, version, super, A, d90, ...}}, subs{}, statics{}, pools{}, version, _state (B02 private)}. Copy-on-write: every assignment change is a new dict with version + 1 (consumers cache on id + version).
- model.groups@(s, __system__): B06.
- model.density@(s, e|class:<rid>): B06. Fields {mu, U_k, lam, n, k, chol_cache}.
- model.rhythm@(s, e|class): B07.
- model.vocab@(s, e|class:<rid>|__system__): B08.
- model.client@(s, e|__system__): B09. The __system__ model also carries the class tiers 'classes' {'class:<rid>': {c, N, members}} (contract C has no class key for model.client), the UA / JA3 co-occurrence table 'cooc' {'family/major': {ja3n: c}} with 'ua_N', and the live rollout tables 'acq' {token: {entity: ts}} and 'known' {entity: ts}.
- model.seq@(s, e|class:<rid>|__system__): B10. Fields {ppm, session_gap, entropy_rate}.
- model.timing@(s, e): B11.
- model.beacon@(s, e): B12.
- model.budget@(s, e): B13.
- model.cp@(s, e): B14.
- model.identity@(s, __system__): B15. Fields {fmt, version, fitted_ts, entities, roles, pca, W, means, class_means, class_var, bg, llr_calib{m: (a, b)}, llr_n, confusion, anonymity_sets, stats{e: {recall1, recallK, eer_hard, t99, separability, near, n_windows, confusable_with}}, classes, distinctive, modality_share} (lib/m_identity docstring). EER_hard is the max pairwise EER over the 3 nearest impostors. Collection runs every tick; the fit every 96 ticks or 24 h.
- model.idwin@(s, __system__): B15.
- model.link@(s, __system__): B17. Fields {links: [{id, from, to, ts, conf, status, retracted, retracted_ts, rollback_to, reason, ...}], actors: [{id, members, links, first_ts, last_ts}], version, shared (private), pending (private)} (lib/m_link). A retracted link keeps its record; B28 answers it with model.control {rollback_to: t_link, release: [t_link, now]} on the seeded entity.
- model.classagg@(s, class:<id>): B18.
- P2 models: model.mixture@(s, e) (B19), model.session@(s, e) (B20), model.xsys@(__org__, __org__) (B21), model.embed@(s, __system__) (B22).
- model.feedback@(__org__, __org__): B23. Fields {family_w, detector_prec, policies, allowlist, alpha_mult, accept[(s, e, ts)], freeze[(s, e, ts)], queue[]}.
- model.calib@(s, e): the detector rings (key (d, stratum)) and GPD fits are owned by B24; the meta rings (keys 'meta_inst', 'meta_all') are owned by B25.
- model.control@(s, e): B28. Fields {version, branch, rebase_from, rollback_to, release[t0, t1], frozen, allow_drift, accepted_class_change}.
- model.governor@(s, e): B28. Fields {regime, onset, type, logodds, evidence, history}.

Checkpoints: store.put_checkpoint(s, e, learner, ts, blob) and get_checkpoint(s, e, learner, at_or_before). Retention is geometric at ages of {1, 2, 4, 8, 16, 32, 64, 128, 168} h, with at most 10 per key.

D. MetricStore API additions (backward compatible)

- add_vec(s, e, name, ts, arr), vec_tail(s, e, name, n), vec_since(s, e, name, since), vec_at(s, e, name, ts). derived_series and latest_derived on a virtual name return DerivedMetric views.
- raw_tail and derived_tail use islice on a reversed deque. This replaces the full copy in store.py:80-86.
- latest_fresh(s, e, name, now) returns the value only if ts == now. Also latest_raw_at(s, e, name, ts).
- first_seen(s, e) and last_seen(s, e) are maintained in add_raw(m, touch=True).
- entities(s, include_pseudo=False), pseudo_entities(s), entities_active(s, since).
- events(system=None, entity=None, since=None, kinds=None, limit=200) and matches(system, entity, since=None, categories=None, limit).
  - Both use per-(system, entity) and per-system indexed deques, bisected on ts: O(log n + k). This replaces the linear scans at store.py:131-145.
- add_event(e) returns an id; get_event(id); update_event(id, **kw).
- add_label and labels(s, e, since).
- put_incident, incidents(s, e, status, since), get_incident(id).
- put_profile_version and profile_versions(s, e, n=12).
- put_model, get_model, model_version.
- put_checkpoint and get_checkpoint.
- set_retention and memory_report(). ensure_retention(prefix, max_points=None, max_age_s=None) is the raise-only form for inputs several engines need (D0 / D2 on act.events): it never lowers a rule the store default or another engine set.
- put_health(engine, dict), health(), last_write_ts(s, e, name).
- timeline(s, e, since): merges indexed events, matches, incidents, profile versions and risk points.
- snapshot(s, e, now=None, names=None): when now is given, it returns only fresh values; with names, only those metrics (the same values; lib-4 passes the metrics its signatures reference, since lib-3 adds ~300 derived names and vector views per entity).
- RawMetric, DerivedMetric, BehaviorEvent and SignatureMatch become dataclass(slots=True) with None defaults for dims and inputs.

E. Schema additions

- BehaviorEvent gains: id, status ∈ {open, suppressed, acked, closed}, p_value, e_day, axes, p_by_detector, dedupe_key, incident_id, model_version, window.
- Novelty events (first_seen, rare_access) carry extra {dim, value, tier ∈ {entity, class, system}, bits, idf, adopted, discount, flags {sensitive, admin, external, upload_dominant, new_external_domain, low_prevalence}} with the flags also as top-level keys; B05 (system-tier exclusion, read at t-1) and B26 (weight, repeat key, stage) read them.
- Label(id, system, entity, target_type ∈ {event, incident, entity, class}, target_id, verdict ∈ {tp, fp, expected_change, benign_known, unsure}, scope ∈ {this, pattern, entity, class, system}, t0, t1, ttl_s, analyst, note, ts).
- Incident(id, system, entity (may be class:<id>), entities, kinds, axes, status, opened, last_seen, severity, e_day_min, risk, evidence, explanation, narrative, campaign_id, parent_id, close_reason ∈ {returned, accepted, labelled, timeout}).
  - A common-mode child is status 'suppressed' with parent_id set (and an evidence entry state 'suppressed_common'); there is no separate 'suppressed_common' status.
  - The quiet close (no alarm or finding for max(8 ticks, 2 h), every accumulator below h/4 on the calibrated-p scale, q_inst quiet) uses close_reason 'timeout'. Nothing is closed by age.
- EntityProfile.separability = clip(1 − 2·EER_hard, 0, 1).
- EntityProfile.stable = n_eff ≥ 96 and calibration healthy.

F. Event kinds

- incident: extra.state ∈ {open, escalate, update, close}. The entity may be a class.
- Novelty: first_seen, rare_access, class_adopted.
- Client: client_change, client_impersonation.
- Identity: identity_mismatch, unknown_identity, low_identifiability.
- Linking: entity_resolution, possible_impersonation, shared_ip, identity_moved, link_retracted.
- Classes: new_entity_matched, new_entity_unmatched, class_transition, class_split, class_merge, peer_outlier.
- Coherent shifts: system_shift, coherent_shift, class_shift, class_adoption_risky.
- Temporal and cumulative: schedule_shift, beacon, budget_exceeded, baseline_creep.
- regime: state ∈ {suspect, drifting, returned, accepted, rejected, rollback}; type ∈ {intensity, shape, categorical, rhythm, ramp, identity, new_entity, c2, exfil}.
- pipeline_degraded.
- The legacy kinds anomaly, drift and sequence are no longer emitted. See api_ui_spec for the compatibility projection.

G. profile.extra keys

- maturity; model_state (predictive p5/p50/p95 in natural units for the current bucket); categorical; client_stacks; rhythm; timing; beacons; sequence; budget.
- peer_group {role, role_name, sub, prob, static_classes, pool, class_path}.
- identity {recall1, recallK, eer_hard, t99, confusable_with, distinctive, modality_share}.
- attribution; continuity {continuity_id, aliases, linked_from, linked_to, entity_kind, shared_ip}.
- risk {score, tier, trend, top_reasons, stages}.
- regime {version, branch, state, onset, type, p_legit, history, rollbacks}.
- calibration; feedback; class_monitor (class profiles only); portrait {json, text_zh, text_en, version, diff}; mv_model; health.

H. Gating contract (lib/gating.py)

- D_ticks = max(4, ceil(D_min_s/Δt)), with D_min_s = 600.
- commit_candidates(store, s, e, learner, last_ts, now) returns [(ts, w_eff, w_prov)]. A row is committed only if quarantine at t−1 is 0 and the learner is not frozen; otherwise it is appended to held.
- apply_control(state, control):
  - rebase_from: replay held rows with ts ≥ rebase_from, weighted by w_prov, into version+1.
  - rollback_to: restore the checkpoint at or before rollback_to, replay committed rows in (ckpt_ts, rollback_to] with their recorded w_eff, and hold everything after.
  - release[t0, t1]: commit the held rows in [t0, t1] with w_prov.
  - frozen: stop all commits.
  - allow_drift: sets the reference anchor's permitted slope (accepted legitimate ramp).
- Checkpoints are written at most hourly; the reference anchor is checkpointed daily.
- Every learner honours this contract: B03, B06–B14, B15 (window buffer), B18, B19, B20, B24 (ring entries carry ts), B25 (meta rings).
- Training: the governor writes trust = trust_prov = 1, unless a lib-4 match of HIGH or above exists at that tick.
- Link seeding: when model.link.version increases, each learner seeds B := B_own + 0.5·A. This covers baseline, rhythm, vocab, client, seq, timing, budget, density, calibration rings (A's most recent M/2 entries per stratum) and identity.

I. Context and config

- ctx.window_s is the actual Δt.
- ctx.config keys:
  - tz (default Asia/Shanghai);
  - calendar {holidays[], makeup_workdays[]};
  - ip_classes [{name, cidrs, systems, criticality}];
  - dhcp_scopes; sensitive_patterns; org_domains;
  - alert_budget {entity_per_hour: 3, system_per_day: 20};
  - strict (bool); daypart_day_hours (8–20); D_min_s (600);
  - budget_abs_floor {Q: natural units per horizon} (B13; each key overrides the engine default: bytes_up 5 MB, bytes_down 50 MB, writes 200, slots 8, up_novel 20 MB, dns_label 200 KB, objs 500, templates 50, dests 50).

J. Detector registry (lib/detectors.py)

DETECTORS = [marg_int, marg_shape, peer, t2, spe, offhours, silence, novelty, novelty_rate, jsd, client, seq, dwell, timing, beacon, budget_vol, budget_exfil, budget_breadth, cusum, mcusum, bocpd, creep, identity, class_int, class_shape, class_rhythm, class_novel, class_coherence, mixture, session, cross_system].

Each detector has: family; kind ∈ {inst, acc}; default axes; owner engine; strata (identity uses regime strata); wall-clock alarm budget (for accumulators). The table is in architecture_md §4.

K. Stage map (lib/stages.py)

Axis, event or lib-4 category to stage:
- volume, shape, peer → behavior
- temporal → off_hours
- categorical: sensitive or admin → privilege; external upload-dominant → exfiltration; new external domain or low prevalence → c2
- exfil → exfiltration
- breadth: internal peers or ports → discovery; object ids → collection
- sequence with an auth token family or 4xx on login → credential
- identity, client_impersonation, possible_impersonation → identity
- beacon → c2
- xsys → lateral
- lib-4 categories: recon/scan → discovery, auth/bruteforce → credential, tunnel/beacon/c2 → c2, transfer/exfil → exfiltration, admin → privilege

L. Class keying

- Role ids are global (org-wide clustering). Class state is instantiated per system under the key (s, 'class:<rid>').
- Vocab, PPM, rhythm, density and classagg models are always per (system, class).
- Cross-system pooling happens only in B02's role descriptors.
- A class with fewer than 3 members in a system backs off to the system tier, never to org.
- Static classes and pools use 'class:static:<name>' and 'class:pool:<cidr>'.

M. Engine core

- Engine.period_s: the engine runs when its tick interval OR its wall-clock period has elapsed, with a per-entity random phase for per-entity refits.
- safe_run records error_count, last_error_ts and the head of the traceback, and calls put_health. In strict mode it re-raises.
- A consumer whose required input has last_write_ts < now, or whose producer has an error at this tick, writes NaN plus behavior.degraded.
- If more than 30% of an entity's families are degraded, a pipeline_degraded system event is emitted.