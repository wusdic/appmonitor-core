PREREQUISITES (pipeline/generator.py v2 and pipeline/build.py)

1) Persona v2 = Persona(entity, archetype, params). params are drawn deterministically from crc32(system|entity):
- Volume scale ~ LogNormal(0, 0.35).
- Work window: start ~ N(8.5 h, 0.7), end ~ N(18 h, 1.0), local time. Lunch dip 12–13 at 0.4. Weekday mask.
- Path preference ~ Dirichlet(α = 0.4) over a vocabulary of 30–40 templates per system. The vocabulary includes private paths: /fin/ledger, /hr/leave, /crm/lead, /wh/stock, /report/{name}, /orders/view/{id}.
- /hr/salary/* is used only by the HR persona 10.20.1.21.
- A navigation Markov chain (login → dashboard → list → view{id} …) with personal transition weights.
- A personal object-ID range. Error rate ~ U(0.01, 0.08).
- Device tuple: 70% use the org standard (Chrome/126, Windows, JA3_A, TTL 128); 30% are unique (Edge, Firefox, or Safari-mac with TTL 64).
- Think time ~ LogNormal(ln 8 s, 1.0). RTT per subnet ~ U(3, 45) ms. Response-size multiplier ~ LogNormal(0, 0.35).
- The twin fixture 10.30.2.27/.28 shares 90% of its parameters, so it is the confusable pair.

2) Machine personas:
- API clients: an endpoint subset, a client-library UA (python-requests/2.31, okhttp/4, Go-http-client/1.1), their own JA3, a poll period with jitter, and a batch size.
- Integration: every 5 min.
- Health: every 30 s ± 1 s.
- Backup: daily 02:00 ± 10 min for 40 min, except 10.20.9.5, whose window is 01:00.
- NTP-like pollers: on the health hosts.
- NAT host 10.30.2.50: carries 2 interactive personas throughout.

3) Mechanics:
- Counts ~ Poisson(rate·dt/60).
- Per-event obs.ts is a real time. Human sessions are a continuous-time process independent of the tick grid (eval round 2): session starts are Poisson with rate rate·activity/sess_len per 5-min sub-interval, a session holds Geometric(1/sess_len) events with log-normal think times, and events after the tick end are carried to the tick that contains them. (Sessions used to be laid out inside each tick, so at dt = 60 s every minute started a session: ~4x the login redirects and DNS lookups, ~2.5x the POST share and a median inter-request gap of 50 s instead of 9 s, relative to 900 s.) A scenario's forced activity window holds at least one session. Machine resolver clocks run on every tick (a 90-150-s poller used to skip its DNS lookups on ticks without a poll), and a backup job resolves once per run, not once per tick.
- Idle ticks are real; no DNS query is forced.
- Calendar: weekends, a holiday list and 调休 make-up days, from ctx.config.
- Attack modes are additive or replace, each with a start and an end.

4) Aggregated mode, used when dt ≥ 900 s:
- One Observation per (entity, token, outcome, dest, stack), with extra = {count, bytes_up_total, bytes_down_total, retransmits_total (HTTP), ts_sample: up to 64 offsets}. Per-record fields are per-flow values (R1 adds w × field); totals travel in extra.
- Raw engines honour count (R1, R2).
- This cuts generator plus raw cost from ~10.6 µs per event to ~10.6 µs per aggregate, about 30 aggregates per entity per tick.

5) Clock and truth:
- Pipeline.run_tick(obs, now = gen.vt, dt) sets ctx.window_s = dt. The live loop uses gen.vt (fixes build.py:114-115).
- gen.truth = [{scenario_id, pack, system, entities, t_start, t_end, label ∈ {malicious, legit_change, system_change}, expected_detectors, expected_axes, max_ttd, required_severity | max_allowed_severity, perturbed_features}].
- Scoring code never reads gen.truth.

POPULATION (39 base entities; 1–8 extra per pack)
- erp-prod (10.20.x):
  - Interactive: 1.11, 1.12, 1.13, 1.15, 1.16, 1.17, 1.18; HR persona 1.21.
  - API: 4.30, 4.32, 4.33. Integration: 4.31. Backup: 9.5. Health: 9.9.
  - Reserved new IPs: 1.14, 1.66, 7.77, 1.70–1.72, 1.112, 1.114.
- oa-portal (10.30.x):
  - Interactive: 2.21, 2.22, 2.24, 2.25, 2.26, 2.29, plus the twins 2.27 and 2.28.
  - Search: 2.23. API: 4.40, 4.41, 4.42. Health: 9.9. NAT: 2.50.
  - Reserved new IP: 2.99.
- api-gateway (10.40.x):
  - API: 4.51, 4.52, 4.54, 4.55, 4.56, 4.57. Integration: 4.53. Health: 9.9. Backup: 9.6.
- The smoke persona set is the 17 original entities plus 1.15, 2.24 and 4.54, 20 in total.

PACKS
Rule: one scenario per entity per pack. Class-wide legitimate scenarios get their own packs. Timezone and cadence are per run. 5 seeds per pack.
- Pack A, short/identity. Asia/Shanghai.
  - Warm-up: 14 d at 3600 s aggregated (336 ticks), then 2 d at 900 s (192).
  - Scenario phase: 4 d at 900 s (384), so 912 ticks in total; ≈ 2.1 min per seed projected.
- Pack B, long-horizon. Asia/Shanghai.
  - Warm-up: 528 ticks as in Pack A.
  - Scenario phase: 12 d at 900 s (1152), including a weekend, a holiday and a 调休 Saturday. 1680 ticks; ≈ 3.9 min.
- Pack C, class-wide legitimate changes and DST. Europe/Berlin.
  - Warm-up: 528 ticks, ending 2 d before the DST switch.
  - Scenario phase: 7 d at 900 s (672). 1200 ticks; ≈ 2.8 min.
- Pack D, slow legitimate changes, month and class threat. Asia/Shanghai.
  - Warm-up: 528 ticks.
  - Scenario phase: 30 d at 3600 s (720), covering a month end and a holiday. 1248 ticks; ≈ 2.9 min.
- Pack E, cadence switch. Asia/Shanghai.
  - Warm-up: 7 d at 900 s (672).
  - Live: 1 d at 60 s, non-aggregated (1440). 2112 ticks; ≈ 3.9 min.
- CPU projection: lib-3 ≈ 94 ms plus lib-1/2 ≈ 20 ms plus lib-4 ≈ 15 ms plus generator ≈ 10 ms per tick, for about 45 entities.

THREAT SCENARIOS
Onsets are in scenario-phase ticks at the pack cadence, or as day and local time. Each entry lists the expected detectors or axes and the required severity by the deadline.

- T1 (A) 10.40.4.52: beacon repurposing.
  - Onset: tick 40 (workday 10:00). Replace mode: 3 GET /ping per 15 min to c2.example.net.
  - Detected by: B04 marg_int (NB lower tail: 22 → 3 gives p ≈ 5.7e-7), B08 novelty (system-tier SNI), B16 attribution (looks like the health role), B14 bocpd.
  - Requirement: HIGH (corroborated volume + categorical) within 2 ticks. Quarantined until the end. Baseline unchanged.
- T2 (B) 10.20.1.12: low-and-slow exfiltration.
  - Onset: day 2 09:00. Additive, business hours only: POST to ext-store.example.net, 200 KB per 15 min, growing ×1.6 per day for 5 days. Per-tick |z| < 2.5 on days 1–2.
  - Detected by: B08 novelty (system tier, axis exfil) on day 1; B13 budget_exfil on day 2; B14 cusum (bytes_up, updown_log, against the reference); B26 risk ≥ High by day 3 (stages exfiltration, c2 and behavior).
  - Measured after the rollback: the current anchor's bytes_up is ≤ 0.3 log1p from pre-onset; the reference is ≤ 0.1σ15.
- T3 (B) 10.20.4.30: frog-boil.
  - bytes_up +15% per day for 10 days, to the same destinations.
  - Detected by: B14 cusum k = 0.25 against the reference (expected day 2, cumulative ≈ 1.3×), creep, B13 budget_vol.
  - Requirements: an alarm before cumulative 3×; ≥ LOW by day 2 and ≥ MEDIUM by day 4. Never accepted (slope 0.14 per day > 0.05). Reference unchanged within ±0.1σ15.
- T4 (B) 10.30.4.41: duty-cycled exfiltration, 20× upload on 30% of ticks chosen at random, to an existing destination.
- T4b (B) 10.20.4.33: the same on alternate ticks only.
  - Detected by: B14 cusum, B13 budget_vol, B06 t2.
  - Requirement: ≥ LOW within 12 ticks.
- T5 (A) 10.30.2.22: off-hours activity.
  - 02:00–04:00 local on 3 workdays, plus Sunday 14:00–16:00, using its usual paths and volume.
  - Detected by: B07 offhours (alarm on the 3rd active slot, 13.9 bits ≥ 13.7). Corroborated by B04 marg_int in the night bin.
  - Requirements: ≥ MEDIUM by the 4th active slot of night 1. Daytime scores unchanged.
- T6 (A) 10.20.1.11: concurrent client.
  - From tick 20, a python-requests/2.31 client (new JA3, TTL 64) at similar volume, while the user keeps working.
  - Requirement: B09 client_impersonation HIGH within 2 ticks, plus B08.
- T6b (A) 10.20.1.18: spoofed UA.
  - From tick 20, the exact Chrome/126 Windows UA with a Linux JA3 and TTL 64.
  - Requirement: B09 inconsistency I = 1, HIGH within 2 ticks.
- T7 (A) 10.20.1.13: rare sensitive resource.
  - From tick 30, 1 GET /hr/salary/export per tick at normal size (df = 1 of 10).
  - Detected by: B08 rare_access at class tier, bits ≥ 12, axis privilege; B10 surprisal.
  - Requirement: ≥ MEDIUM on the first access.
- T8 (B) 10.20.1.15: slow record enumeration.
  - Day 3 09:00–17:00: /orders/view/{id} with sequential ids at the persona's normal rate, about 2000 distinct ids versus the usual 40 per day.
  - Detected by: B13 budget_breadth (HLL) within the day; B11 burstiness shift.
  - Requirement: marg p > 0.01 on ≥ 90% of ticks (per-tick detectors stay quiet).
- T9 (A) 10.30.2.26: impersonation of a known individual.
  - From tick 30, its traffic is replaced by the exact persona of 10.30.2.21 (templates, rhythm, stack), while .21 continues.
  - Requirement: B16 identity_mismatch with looks_like = 10.30.2.21 and posterior ≥ 0.9 within 4 active windows; HIGH, because .21 is concurrently active. Also B10 class_llr.
- T9b (A) 10.30.2.29: unknown individual.
  - From tick 30, replaced by a new persona drawn from the interactive prior (new seed, new stack, new path preferences).
  - Requirement: B16 unknown_identity at MEDIUM ('same class, different individual'); no class_transition.
- T10 (B) 10.40.4.51: slow jittered additive beacon.
  - From day 4, keeps about 22 requests per tick and adds 1 request of about 1.2 KB to cdn-upd.example.net every 300 s ± 30% (sub-tick timestamps).
  - Detected by: B12 beacon, renewal LRT p < 1e-6 after 12 events (about 1 h); B08 novelty.
  - Requirement: ≥ MEDIUM within 16 ticks.
- T11 (A) cold-start attackers, from tick 50: new 10.30.2.99 (scanner), 10.20.7.77 (DNS tunnel), and 10.20.1.66 (interactive-like plus 2 MB per tick to ext-drop.example.com from its first tick).
  - Detected by: B02 new_entity_unmatched (.99, .77) within 3 ticks; B04 under backoff from tick 1; B08 system tier; B13 budget_exfil for .66 within 8 ticks.
  - Requirements: risk ≥ High. After 50 ticks the three are still quarantined with 0 committed baseline rows.
- T12 (A) 10.30.2.25: low-rate credential stuffing.
  - From tick 60, adds 6 POST /login per 15 min, 90% of them 401 (about 15 requests per tick in total).
  - Detected by: B04 beta-binomial on 4xx and write ratio (p < 1e-5); B10 dwell in auth (stage credential).
  - Requirement: ≥ MEDIUM within 3 ticks.
- T13 (A) 10.20.1.16: privilege misuse at unchanged volume.
  - From tick 80, 60% of requests move to /admin/users/{num} and /admin/export.
  - Detected by: B08 class/system tier (privilege), B06 spe, B04 marg_shape. The intensity channel stays quiet; the portrait diff lists the paths.
  - Requirements: ≥ MEDIUM within 2 ticks. Never auto-accepted.
- T14 (B) 10.20.1.13: DNS tunnel on an existing host.
  - From day 3, 2 TXT queries per tick with 30–48-character labels under a new eTLD+1, tun.example.org.
  - Detected by: B08 (qtype, system-tier domain), B13 budget_exfil (DNS label bytes), B04 dns_txt_ratio.
  - Requirement: ≥ MEDIUM within 4 ticks.
- T15 (B) 10.20.9.5: baseline-poisoning ramp, then strike.
  - From day 4 01:00, uploads grow 3% per tick (compounding) for 200 ticks. An external SNI is introduced at ramp tick 50. A 30× strike follows at ramp tick 200.
  - Detected by: B14 cusum/creep against the reference; B08 at tick 50; B28 SUSPECT with rollback. The strike is judged against the reference/golden anchor.
  - Requirements: HIGH within 1 tick of the strike. Never accepted.
- T16 (A) 10.30.2.24: several weak signals.
  - Within 6 h from tick 120: a JA3 minor variant, one rare path at class tier, a persistent +0.7σ bytes shift, and one off-hours active slot. Each is below its own alarm.
  - Detected by: B25 evidence plus B26 stage diversity.
  - Requirements: risk ≥ High within 6 h, with an incident ≥ MEDIUM.
- T17 (A) 10.40.4.53: quiet role repurposing.
  - From tick 140, stops POST /api/sync and starts GET /v1/resource at the same volume.
  - Detected by: B02 class_transition (3 runs); B16 mismatch at class level (looks like the api role); B14 shape.
  - Requirement: MEDIUM (via the self/peer 2×2).
- T18 (B) 10.20.1.16: correlation-break exfiltration.
  - From day 7, for 48 ticks, bytes_up per request is ×3–4 while counts and paths are unchanged; every marginal is below p95.
  - Requirement: B06 spe p < 1e-4 with RBC {bytes_up, updown_log, bytes_per_flow} within 6 ticks.
- T19 (B) IP-hopping actor.
  - 10.20.1.70 → .71 → .72, 30 ticks each from day 9: interactive-like plus 1 MB per tick to ext-sync.example.io.
  - Detected by: B17 actor chain (2 links); B13 actor-level budget_exfil; B08 (the same rare destination from new IPs).
  - Requirement: one actor incident within 60 ticks.
- T20 (D, P2) 10.30.2.26: lateral access.
  - From day 12, accesses api-gateway /v1/resource and erp /admin.
  - Requirement: B21 first_access_system at role tier, stage lateral, ≥ LOW.
- T21 (D) compromised pool in api-gateway.
  - 4 of 6 API members (10.40.4.54–.57), staggered over 2 h from day 20 10:00, add a POST of about 30 KB per 15 min to telemetry-sync.example.org.
  - Per entity: first_seen is discounted to INFO by adoption, and per-entity budgets stay below the floor. No per-entity incident is required.
  - Requirements: B18 class_novel (external, upload-dominant) plus class_shape give a class incident ≥ MEDIUM within 4 ticks of the 3rd adopter. Class risk ≥ Medium. The class aggregate baseline is not absorbed.

LEGITIMATE CHANGES
Each must not alert above the stated level.

- L1 (D) month-end: the erp interactive role (1.11–1.18, 1.21) runs at ×2.5 volume with the same mix on the last 2 business days.
  - Allowed: at most one coherent_shift class incident ≤ LOW; 0 member incidents ≥ LOW.
  - Member and class risk stay < 30.
- L2 (C) client rollout: the oa interactive role moves from Chrome/126 to 127 (new JA3) on one day.
  - Allowed: class_adopted and client_change at INFO only; no impersonation.
  - Class version bump.
- L3 (C) new resource: the erp interactive role starts GET /v2/orders on one day.
  - Allowed: class_adopted INFO. Entity novelty ≤ INFO. class_novel p > 0.01.
- L4 (B and D background) calendar: weekends, a holiday and a 调休 Saturday.
  - No offhours or silence events. No incident ≥ LOW attributable to calendar days.
- L5 (D) 10.30.4.40: grows 2% per day for 30 days.
  - Allowed: at most one regime LOW, then accepted (ramp slope 0.02 ≤ 0.05).
  - Never ≥ MEDIUM.
- L6 (A) DHCP renumbering: 10.20.1.12 moves to .112 at tick 100; .12 goes silent.
  - Required: entity_resolution with conf ≥ 0.9 within 4 active ticks of .112.
  - No new_entity_unmatched. At most 1 entity-tier novelty event. Risk < 30.
  - Negative control: a new persona on .114 at tick 100 is not linked and gets only new_entity_matched INFO.
- L7 (all) 10.30.2.50: NAT with two personas.
  - Allowed: shared_ip INFO. No HIGH.
- L8 (A) 10.20.1.14: new employee at tick 70.
  - Required: new_entity_matched with prob ≥ 0.8 within 3 ticks. Nothing else.
- L9 (D) 10.20.9.5: backup moves permanently from 01:00 to 03:00 from day 8.
  - Allowed: schedule_shift LOW (silence and off-hours explained), accepted after 3 days.
  - Never ≥ MEDIUM.
- L10 (C) api-gateway outage and WAN degradation.
  - 5xx for all clients for 3 ticks on day 2 at 11:00; RTT ×3 and retransmits ×5 for 12 ticks on day 3.
  - Allowed: one system_shift or coherent_shift per event. 0 entity incidents ≥ LOW.
- L11 (all) idle and near-idle nights, including empty ticks.
  - 0 incidents. Ratio features NaN or p > 0.05.
- L12 (D) 10.30.2.21: absent for days 5–12, then returns.
  - No silence alert (human rhythm). No link. No offhours on return.
- L13 (C) Europe/Berlin across the DST switch on day 3.
  - No alerts.
- L14 (D) 10.30.2.24: browser auto-update on day 6 (UA minor version and JA3 change; the old stack disappears).
  - Allowed: client_change INFO only.
- L15 (all) sanctioned periodic automation: health, backup and NTP-like polling.
  - No beacon events. Risk < 30.
- L16 (D) 10.30.2.29: explorative user with about 5 new templates per day, all common among peers.
  - No first_seen above entity tier. No incident.

PACK E REPLAYS (60-s cadence)
- T1' on 10.40.4.52, T5' on 10.30.2.22 and T12' on 10.30.2.25, with the same semantics expressed in wall time.
- Every other entity is a clean control.
- Purpose: check that cadence invariance holds for false-alarm rate (FAR) and time-to-detect (TTD).

FIXTURES
- The twins 10.30.2.27 and .28 are present in every pack for the identification gate.