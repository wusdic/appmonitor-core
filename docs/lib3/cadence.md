# Cadence-invariant behaviour library (lib-3 v2.1)

Status: **implemented (M0–M8)**; `grain_mode='canonical'` is the default for
pipeline, runtime, eval and scripts, `'tick'` (v2, golden-tested) for engine
unit tests. §17 lists where the implementation deviates from or corrects this
design, and what is still open. §17.1 records what evaluation rounds 2–3 changed in
canonical mode and the measured state (integration.md §10, §11). `contract.md` and `helpers_api.md` carry the
matching v2.1 amendments, each marked *spec v2.1*.

Scope: the 52-feature representation (FEATURE_SPEC), everything that learns
or scores it (B03, B04, B05, B06, B14, B18), identity windows (B15, B16, B17),
and everything downstream that turns scores into decisions or displays them
(B24–B30, API, eval, packs, runtime). Tick-native detectors (B07–B13) keep
their own wall-clock or event clocks. §9.6 lists what they still depend on.

Notation: Δt is the tick length (`ctx.window_s`). G is a grain length:
G_h = 3600 s, G_q = 900 s, and m = G_h/G_q = 4. "H row" and "Q row" are
feature rows aggregated over a trailing window of length G_h or G_q. The
"T stream" is everything scored per tick. B13's `'<Q>.<H>'` budget keys
(quantity.horizon) are unrelated to these grains.

---

## 0. Decisions at a glance

The lead's direction (points 1–5) is kept. The points below make it precise.
Where a point refines the direction, the reason is given here and expanded
in the section cited.

| # | Decision | Why (section) |
|---|---|---|
| D1 | Every scored feature is defined on a canonical trailing wall-clock window: H (3600 s, primary, observable when Δt ≤ 3600) and Q (900 s, sensitivity, observable when Δt ≤ 900). Values are built from additive per-tick parts, mergeable set sketches, mergeable count maps, and 15-min-slot span descriptors. They are exact whatever the Δt ≤ G (§2–§4). | This is the lead's point 1. |
| D2 | **Refinement.** B01 writes rolling rows every tick (`feature.live.<g>`), as the lead asked. Rows that are learned and scored (`feature.nat.<g>`, `feature.vec.<g>`, `feature.meta.<g>`) are written only on the grain's **decision ticks**. A decision tick is the first tick at or after a multiple of G, so consecutive scored rows never overlap. | A rolling row at Δt = 60 overlaps its predecessor by 59/60. Learning or scoring every rolling row would weight each hour 60 times, and sequential charts would double-count it. The two kinds of row hold the same content on decision ticks. Keeping them in separate series stops the 8-day learning rings from growing 60× (§2.3, §7.1). |
| D3 | **Refinement.** Grain boundaries are epoch-aligned and the same for every entity. There is no per-entity phase. | B05 (leave-one-out common mode) and B18 (class aggregates) need every member's row on the same tick. The cost is a CPU spike on boundary ticks (§15). |
| D4 | **Finding, fixed here.** A row's seasonal bucket is the **midpoint** of the interval it covers, not its end stamp. | Today B03 and B04 bucket a row at its tick's end stamp. A 3600-s tick covering 11:00–12:00 and a 900-s tick covering 11:45–12:00 both go to hour 12.0, but their content is centred at 11:30 and 11:52. So warm-up rows (3600 s) and live rows (900 s) in the same bucket describe times 0.4 h apart, and at 3600 s the bucket is 0.5 h late. The error is largest at the start and end of the work day (§2.4). |
| D5 | B03 keeps a full two-anchor H baseline and a native Q **current** anchor. Before native support, the Q predictive is the transferred H predictive, entered as κ_T = 16 **conjugate pseudo-rows**, so native rows take over smoothly. The Q "reference" predictive is always the transferred H reference. | This makes the lead's "transfer, then native" precise without a hard switch. Q keeps the dual-anchor creep protection at no extra state (§6). |
| D6 | The transfer factor is v_f = 1 + ((m−1)/m)·ω_f. ω_f is the ratio of the within-hour variance of quarter values to the H-grain (excess) variance, estimated on paired hours and EB-shrunk entity → class → system → org → default. The default ω = m gives v = m, the independent-quarters bound, which is conservative under non-negative within-hour correlation. For NB it scales only the overdispersion term, so it is overdispersion-based as the lead asked. | The formula is exact under exchangeable quarters. It was checked numerically: v̂ = 4.005, 1.694, 1.097, 1.000 against true ratios 4.005, 1.694, 1.097, 1.000 at intra-hour correlation 0, 0.5, 0.9, 1 (§6.3). |
| D7 | Sequential charts use **decimated non-overlapping sampling** at the grain period (the lead's first option). Thresholds come from ARLs counted in grain periods. | The AR(1) alternative was rejected. Overlap makes the rolling series MA(m−1) with lag-k correlation (1 − k/m)₊, not AR(1). An AR(1) prewhitener under-corrects at lags ≥ 2, and the right correction would depend on Δt through m. Decimation is exact, costs 1/m of the updates, and makes h independent of Δt by construction (§7.2). |
| D8 | Detectors belong to **streams**: H, Q or T. The single-tick budget is split over **tick types** (h, q, t) with fixed shares β = (0.5, 0.25, 0.25), renormalised over the types present: `e_day = q_all · n_τ / β_τ`. There is one evidence CUSUM per stream (S_h, S_t), each with half the evidence budget. Q evidence never enters a CUSUM. | The e_day budget stays exact in total (0.03 per entity-day) at every Δt. The formula reduces to today's `q·86400/Δt` at Δt = 3600. Unlike a split proportional to tick count, fixed shares stop Δt = 60 from starving the hourly evidence of budget (§7.3–§7.4). |
| D9 | **Refinement.** An identity window is **4 active H rows**: 4 h of wall-clock data at any cadence. B16's identity CUSUMs step only on non-overlapping windows. | This is the lead's "wall-clock span" with a fixed sample size. B15's held-out T99 and the χ² typicality need a fixed number of rows per window, and a pure 4-h span would hold 0–4 active rows for sparse entities (§8). |
| D10 | The four span descriptors (periodicity, timing_regularity, req_per_session, duty_cycle) are scored only on their own span's decision ticks: 6 h or 24 h, local-time aligned. | A 24-h duty cycle scored every hour would enter the H stream 24 times per day (§4, §7.1). |
| D11 | Migration uses **tick mode**: `grain_mode = 'tick'`, the degenerate grain configuration G_h := Δt, with no Q and grain series names resolved to today's names. It reproduces v2 bit-for-bit, so the suite stays green while engines move to the grain API. `'canonical'` becomes the default for pipeline, eval and runtime in the last step (§12). | There is one code path, so no permanent fork. |
| D12 | Packs: when the live cadence is ≤ 900 s, the warm-up ends with a phase **at the live cadence** that contains ≥ 1 full local day of each day type and ≥ 2 days in total. The total span stays 16 d. Pack D (live 3600 s) warms up at 3600 s only. Smoke: 96 × 3600 + 24 × 900 (120 warm-up ticks as before, now covering Wed–Sun). | Q-native support only needs Δt ≤ 900. The tick-native detectors, however, keep cadence-class strata in B24, so the last phase must be at the live cadence itself. This is a pack-definition rule, not seed tuning (§10). |

---

## 1. Problem and root causes

Measured in integration.md §7–§8 (seed 0, strict). Packs warm up 14 d at
3600 s + 2 d at 900 s and score at 900 s. The runtime warms up at 900 s and
goes live at 60 s. A: 1/12 threats detected. B: 1/10. Every control entity
alarms and incidents never close. The same code with a 900-s-only warm-up
detects 7/12, with calibration within ~20× of nominal instead of ~500×.

Root causes, each tied to the part of this design that removes it:

1. **Distinct counts are sub-additive.** `distinct_peers`, `distinct_dports`,
   `distinct_templates` and `ja3_diversity` are per-minute rates of a
   per-tick distinct count. At 900 s they are ~4× their 3600-s value, so
   zr ≈ +1.5 to +2.3 persists for 12 days. The reference anchor moves only
   0.03 σ15 per day. The k = 0.25 CUSUM and MCUSUM bank stays latched on
   ~80 % of control ticks. → Fixed by the H-grain union of per-tick set
   sketches (§3.2), which is exact at any Δt.
2. **Jensen effects on log rates and log averages** of bursty quantities
   (`bytes_*`, `bytes_per_flow`, `updown_log`, latencies). The mean of log
   1-h rates differs from the mean of log 15-min rates, and so do the
   variances. → Fixed by building H values from additive sums (exact), and
   by the Q-grain location and scale transfer (§6).
3. **NB and BB dispersion is grain-specific.** B03 fits one overdispersion
   κ̂ and one concentration φ̂ from rows of mixed exposure. The dispersion of
   a 15-min count is not that of a 1-h count scaled down. → Fixed with
   per-grain anchors (one exposure each) and the ω transfer (§6.2–§6.3).
4. **Tick-count identity windows.** K = 4 ticks is 4 h at 3600 s and 1 h at
   900 s. The runtime smoke run showed 16 open `unknown_identity` incidents
   after 16 live ticks. → Fixed with windows of 4 active H rows (§8).
5. **Tick-resolution window descriptors.** `duty_cycle` counts active
   ticks, so a user active 1 minute in an hour scores 1/1 at 3600 s and
   1/60 at 60 s. `periodicity` is undefined at 3600 s, which has fewer than 8
   bins in 6 h. → Fixed with the 15-min slot grid (§3.3).
6. **Calibration rings keyed by cadence class.** Every stratum is empty
   after a cadence switch, and B24 falls back to model pm. → Fixed with
   rings keyed by grain for grain detectors (§9.1). H rings survive the
   switch.
7. **Row buckets at the end stamp** (D4, §2.4).
8. **Packs whose final warm-up covers one day type**: A covers Sat–Sun and
   B covers Tue–Wed at 900 s, while smoke warms up on a weekend only. → §10.

---

## 2. Grains, clocks and decision ticks (`lib/grains.py`, new)

### 2.1 Constants

```
GRAIN_S      = {'h': 3600.0, 'q': 900.0}      M = 4           EPS = 1e-6
COVER_MIN    = 0.95   # set / map features need cov >= 0.95 G
KAPPA_T      = 16.0   # transfer pseudo-rows in the Q predictive (bucket-feature)
KAPPA_OMEGA  = 24.0   # pseudo paired-hours for the EB shrink of omega / delta
OMEGA_MAX    = 2*M    # omega clip [0, 8]  ->  v in [1, 7]
C_T_MIN      = 5.0    # floor of a transferred Beta-Binomial concentration
BETA         = {'h': 0.5, 'q': 0.25, 't': 0.25}      # single-tick budget shares
EVIDENCE_SHARE = {'h': 0.5, 't': 0.5}                # evidence-CUSUM budget shares
SPAN_S       = {'periodicity': 21600, 'timing_regularity': 21600,
                'req_per_session': 86400, 'duty_cycle': 86400}
MODES        = ('tick', 'canonical')        # ctx.config['grain_mode']
```

### 2.2 Definitions (pure functions of now, Δt and the mode)

- **Observable:** `observable(g, dt) = dt <= G_g·(1+EPS)`. H is observable at
  every supported cadence (60, 300, 900, 3600). Q is observable only at
  Δt ≤ 900.
- **Window of a row ending at `now`:** the ticks with ts ∈ (now − G, now].
  A tick at ts covers (ts − Δt_ts, ts].
- **Coverage:** `cov = Σ Δt_i` over the ticks in the window. This is the true
  exposure in seconds, including the overhang when Δt does not divide G.
  NB exposure is `e = cov/60` minutes.
- **Decision tick:** `decision(now, dt, G) = floor((now+EPS)/G) > floor((now−dt+EPS)/G)`.
  That is, the tick's own interval (now − Δt, now] contains a multiple of G
  (epoch-aligned, the same for all entities; D3). Examples:
  - At Δt = G every tick is a decision tick.
  - At Δt = 900, every 4th tick is an H decision.
  - At Δt = 60, every 15th tick is a Q decision and every 60th an H
    decision.
  - Misaligned ticks (12:07:13, 12:22:13 …) work unchanged. Consecutive
    decision rows are G ± Δt apart, so they overlap by less than one tick.
  - A cadence switch needs no special case. After a 3600 → 900 switch at an
    aligned t0, the next H decision is at t0 + 3600, and its window holds the
    four 900-s ticks.
- **Span decision** for a span feature f (§4): the tick's interval contains a
  **local-time** multiple of SPAN_S[f]. That is 00/06/12/18 local for 6 h,
  and local midnight for 24 h (tz-aware; on DST days the local day has 23 or
  25 h, and the slot count in §3.3 follows it).
- **Tick type:** τ(now, Δt) = 'h' if H decision, else 'q' if Q is observable
  and this is a Q decision, else 't'.
- **Decision ticks per day:**
  - n_h = 86400/max(G_h, Δt);
  - n_q = 86400/G_q − n_h if Q is observable, else 0;
  - n_t = 86400/Δt − n_h − n_q.

  The sum is always 86400/Δt.
- **Stream period of a detector d:** `period_s(d, dt)` is max(Δt, G_h) for
  the H stream, max(Δt, G_q) for Q, and Δt for T.
- **Tick mode** (migration, D11): G_h := Δt, Q is never observable, n_h =
  86400/Δt, `row_tctx` uses the end stamp, and `series(base, 'h') == base`.
  Every formula below then reduces to today's v2 behaviour.

### 2.3 Rolling rows versus decision rows

| Series | Written | Content | Used by |
|---|---|---|---|
| `feature.live.<g>` | every tick where g is observable | nat-unit rolling row over (now − G, now]. Additive features every tick. Set, map and span columns are NaN except on decision ticks, where the row equals `feature.nat.<g>` | API `features[]`, B30 "current", display |
| `feature.nat.<g>`, `feature.vec.<g>`, `feature.meta.<g>`, `feature.expo.<g>` (+ `feature.sketch.h`) | decision ticks only | complete grain row | every learner and scorer |

The presence of `feature.nat.<g>` at ts = now **is** the decision flag.
Staleness (contract M) is judged against `grains.last_decision(now, dt, g)`,
not against now.

### 2.4 Time context of a grain row (D4)

`row_tctx(now, g, dt, config) = timebins.tctx_from_config(now − G_g/2, config, dt)`
is the tctx of the window midpoint: hour_local, dow, day_type, bin48,
bin168, slot and daypart. B03 folds the row at that fractional hour with the
same von Mises spread. B04 scores it at the same bucket. B24 and B25 use its
daypart. In tick mode `row_tctx` is today's end-stamp tctx.

---

## 3. Per-tick additive parts and new inputs

### 3.1 Numeric parts: `feature.part[47]` (B01, every tick, every real entity)

The parts come from the fresh sources through B01's existing getter. A stale
source is a part of 0, never NaN, and an average's sum and weight are both 0
when the average is absent. Every part is additive over ticks, so an H or Q
value is a function of Σ parts over the window.

| # | Part | Per-tick value |
|---|---|---|
| 0 | p_dt | Δt of the tick (s). Σ = cov |
| 1 | p_active | feature.active |
| 2 | p_up | l4.bytes_up |
| 3 | p_down | l4.bytes_down |
| 4 | p_flows | l4.flows |
| 5 | p_http | http.requests |
| 6 | p_dns | dns.queries |
| 7 | p_tls | tls.handshakes |
| 8 | p_events | act.events |
| 9 | p_l3b | l3.bytes_total |
| 10 | p_newpeer | derived.new_peer_count |
| 11 | p_write | http.write_count |
| 12 | p_get | http.get_count |
| 13 | p_4xx | http.status_4xx |
| 14 | p_5xx | http.status_5xx |
| 15 | p_3xx | http.status_3xx |
| 16, 17 | p_lat_s, p_lat_n | http.latency_ms_avg·http.requests; http.requests (0 if the average is absent) |
| 18 | p_resp_s | http.resp_bytes_avg·http.requests |
| 19 | p_req_s | http.req_bytes_avg·http.requests |
| 20, 21 | p_ntr_k, p_ntr_n | act.new_template_ratio·http.requests; http.requests (0 while R2 withholds the ratio) |
| 22, 23 | p_dga_s, p_dga_n | derived.dns_dga_score·n_named; n_named = **derived.dns_dga_named_n** (new, D1) |
| 24, 25 | p_nx_k, p_nx_n | derived.dns_fail_rate·derived.dns_fail_rate.n; derived.dns_fail_rate.n |
| 26 | p_txt | dns.txt_count |
| 27, 28 | p_qlen_s, p_qlen_n | dns.qname_len_avg·dns.queries; dns.queries (0 if absent) |
| 29 | p_weak_k | tls.weak_version_ratio·tls.handshakes |
| 30, 31 | p_hsms_s, p_hsms_n | tls.handshake_ms_avg·tls.handshakes; tls.handshakes (0 if absent) |
| 32, 33 | p_think_ls, p_think_n | **derived.think_log_sum**, **derived.think_gaps** (new, D2): Σ ln(gap) and the count of the within-session gaps D2 already selects |
| 34 | p_pkts | l4.pkts_total |
| 35 | p_retx_k | l4.retransmit_rate·l4.pkts_total |
| 36, 37 | p_rtt_s, p_rtt_n | l4.rtt_ms_avg·l4.flows; l4.flows (0 if absent) |
| 38 | p_syn | l4.syn_count |
| 39, 40 | p_dur_s, p_dur_n | l4.flow_duration_ms_avg·l4.flows; l4.flows (0 if absent) |
| 41 | p_probes | probe.probes |
| 42 | p_reach_s | probe.reachable·probe.probes |
| 43 | p_loss_k | probe.loss_ratio·probe.probes |
| 44 | p_sni_n | derived.sni_entropy_n |
| 45 | p_path_n | derived.path_entropy_n |
| 46 | p_dnsent_n | derived.dns_name_entropy_n |

The weight of an average is the feature's n_source on ticks where the
average exists. When some items lack the measurement (for example requests
without a latency sample), the result is the n_source-weighted mean of the
per-tick means. It is exact when every item carries the measurement, which
is what FEATURE_SPEC's n_source already assumes. Parts are float32 in the
ring, and Σ of ≤ 60 values keeps a relative error below 1e-6.

### 3.2 Set parts: mergeable distinct-count sketches (raw engines, new series)

`SetSketch` (in `lib/sketch.py`, next to HyperLogLog) is a dict value.
- Per tick it is `{'n': distinct, 'h': sorted uint64 list}` when there are
  ≤ 256 keys, else `{'n': distinct, 'hll': 1024 bytes}`.
- Hash: blake2b-64 of `f'{ns}\x1f{key}'`, the same function as
  `sketch.token_hash`. A 64-bit key collides with probability ~n²/2⁶⁵,
  which is negligible.
- `union(sketches)` stays exact while |∪| ≤ 4096 and otherwise converts to
  HLL: register-wise max, lossless merge, standard error 3.25 %.
- `count(sk)` returns the exact size or the HLL estimate.

| Series | Owner | Keys (the full set, before top-64 truncation) |
|---|---|---|
| `l4.peer_ids` | R1 l4flow | peer strings, exactly what `l4.distinct_peers` counts (`a.peers`) |
| `l4.dport_ids` | R1 l4flow | destination ports (`a.dports`) |
| `tls.ja3_ids` | R1 tls | JA3 hashes (`a.ja3`) |
| `act.template_ids` | R2 action_token | `_tkey(t)` over `acc.tokens` (what `act.distinct_templates` counts) |

Retention is 75 min (`ensure_retention(max_age_s=4500)`), which is ≥ G_h +
max Δt of the Q-observable cadences.

### 3.3 Map parts and span inputs

- **Map features** (bounded entropies and concentration) merge the raw
  count maps over the window on decision ticks: `tls.sni_set`,
  `http.top_paths` and `dns.qname_set` (top-64 + `__other__` per tick).
  The named counts are summed per key and `__other__` is summed. D1's rules
  then apply unchanged: entropy over the named entries, n incl. other,
  concentration = top named / total.
- **`feature.sketch.h[80]`** is `sketch.sketch_vector` over the merged maps
  of the five namespaces: act.tokens, client.stack_set, tls.sni_etld1_set,
  dns.qname_etld1_set and l4.dport_set.

  B01 raises the retention of these raw sets to 75 min (raise-only).
- **`act.slot_events`** (R2, new) is `{slot_start_epoch: events}` over the
  15-min slots the tick touches. It counts every event, including aggregated
  records, whose weight is split over their `ts_sample` offsets
  proportionally. At Δt ≤ 900 aligned it has one entry, at 3600 it has four.
  Retention is 25 h (it joins the D0 inputs).
- **D0 and D2 on the slot grid** (spec change, **canonical mode only**;
  tick mode keeps v2's tick grid, and so does a tick whose entity has no
  `act.slot_events`, which falls back to the tick grid):
  - `duty_cycle` = active 15-min slots / covered 15-min slots in the
    trailing local 24 h. A slot is covered when an act.events clock tick
    overlaps it. This is v2's time weighting on a fixed slot grid, so a new
    entity ramps as today: 1 of 1, then 1 of 4.
  - `periodicity_score` and `beacon_lag` = the largest autocorrelation of
    per-slot events over the trailing 6 h (24 bins, lags 2..12, lag reported
    in s). It is now defined at 3600 s too.
  - `timing_regularity` = 1 − CV of per-slot events over active slots in
    6 h (≥ 2 active slots).

  At Δt ≤ 900 these equal a re-binning of the tick grid. At 3600 they come
  from the event timestamps, so they no longer depend on the tick length.
  `req_per_session` is unchanged (stream timestamps, 24 h).
- **`think_time`** at a grain is the geometric mean of the within-session
  gaps: exp(Σ ln gap / n), n ≥ 2. The median is not mergeable. The tick-level
  `derived.think_time_s_avg` (median) stays for lib-4, B02's cold path and
  the tick vec.

---

## 4. Per-feature definition table (all 52)

Columns:

- **Class** says how the grain value is built:
  - `add` = function of Σ parts;
  - `set` = |∪ SetSketch|, which needs cov ≥ 0.95 G;
  - `map` = merged count maps, which needs cov ≥ 0.95 G;
  - `span` = D0/D2 value at the tick.
- **Grain value (nat)** is the natural-unit value over the window: Σ = sum
  over the window, cov = coverage in s.
- **vec** is the FEATURE_SPEC transform with dt := cov.
- **Family**: NB = negative binomial with exposure e = cov/60 min, BB =
  Beta-Binomial on (k, n), t = Student-t on vec from the NIG. The same family
  applies to H and Q. Parameterisation is in §6.2.
- **Q transfer**:
  - `rate`: NB, same per-minute μ, 1/r_Q = v/r_H;
  - `ratio`: BB, same p, 1 + c_Q = (1 + c_H)/v;
  - `jensen`: t, scale² × v, default location shift δ = −(v − 1)σ²_H/2;
  - `mean`: t, scale² × v, default δ = 0;
  - `none`: Q is native-only and unscored until native support ≥ κ_T;
  - `—`: not a Q feature (span, H only).

| # | Feature | Kind | Parts | Class | Grain value (nat) | Family | Q transfer |
|---|---|---|---|---|---|---|---|
| 0 | bytes_up | bytes | p_up | add | Σp_up (bytes per window). vec = log1p(Σ·60/cov) | t | jensen |
| 1 | bytes_down | bytes | p_down | add | Σp_down | t | jensen |
| 2 | flows | count | p_flows | add | Σp_flows | NB | rate |
| 3 | http_requests | count | p_http | add | Σp_http | NB | rate |
| 4 | dns_queries | count | p_dns | add | Σp_dns | NB | rate |
| 5 | tls_handshakes | count | p_tls | add | Σp_tls | NB | rate |
| 6 | intensity | count | p_events | add | Σp_events | NB | rate |
| 7 | bytes_per_flow | avg | p_l3b, p_flows | add | Σp_l3b/Σp_flows (n = Σp_flows ≥ 1) | t (log) | jensen |
| 8 | updown_log | avg (identity) | p_up, p_down, n = p_flows | add | ln((Σp_up+1)/(Σp_down+1)) | t | mean |
| 9 | distinct_peers | count | l4.peer_ids | set | \|∪\|. e = cov/60 inside the gate | NB | none |
| 10 | distinct_dports | count | l4.dport_ids | set | \|∪\| | NB | none |
| 11 | distinct_templates | count | act.template_ids | set | \|∪\| | NB | none |
| 12 | new_peer_count | count | p_newpeer | add | Σp_newpeer. Exact: D1 counts a peer new only at its first sighting in 30 d, so the per-tick counts sum to the distinct new peers of the window | NB | rate |
| 13 | dest_concentration | bounded | tls.sni_set else dns.qname_set; n = Σ(p_tls + p_dns) ≥ 5 | map | top named / total of the merged map (TLS if the window has any TLS) | t (logit) | none |
| 14 | http_write_ratio | ratio | p_write / p_http | add | k = Σp_write, n = Σp_http | BB | ratio |
| 15 | http_get_ratio | ratio | p_get / p_http | add | Σ/Σ | BB | ratio |
| 16 | http_4xx_rate | ratio | p_4xx / p_http | add | Σ/Σ | BB | ratio |
| 17 | http_5xx_rate | ratio | p_5xx / p_http | add | Σ/Σ | BB | ratio |
| 18 | http_3xx_rate | ratio | p_3xx / p_http | add | Σ/Σ | BB | ratio |
| 19 | http_latency | avg | p_lat_s / p_lat_n | add | Σp_lat_s/Σp_lat_n | t (log) | jensen |
| 20 | resp_bytes_avg | avg | p_resp_s / p_http | add | Σ/Σ | t (log) | jensen |
| 21 | req_bytes_avg | avg | p_req_s / p_http | add | Σ/Σ | t (log) | jensen |
| 22 | path_entropy | bounded | http.top_paths; n = Σp_path_n ≥ 5 | map | normalised entropy of the merged named paths | t (logit) | none |
| 23 | new_template_ratio | ratio | p_ntr_k / p_ntr_n | add | Σ/Σ (NaN while Σp_ntr_n = 0) | BB | ratio |
| 24 | dns_name_entropy | bounded | dns.qname_set; n = Σp_dnsent_n ≥ 5 | map | normalised entropy of the merged qnames | t (logit) | none |
| 25 | dns_dga_score | avg | p_dga_s / p_dga_n; gate Σp_dns ≥ 1 | add | Σ/Σ (named-mass-weighted mean first-label entropy) | t (log) | jensen |
| 26 | dns_fail_rate | ratio | p_nx_k / p_nx_n | add | Σ/Σ | BB | ratio |
| 27 | dns_txt_ratio | ratio | p_txt / p_dns | add | Σ/Σ | BB | ratio |
| 28 | dns_qname_len | avg | p_qlen_s / p_qlen_n | add | Σ/Σ | t (log) | jensen |
| 29 | sni_entropy | bounded | tls.sni_set; n = Σp_sni_n ≥ 5 | map | normalised entropy of the merged SNIs | t (logit) | none |
| 30 | tls_weak_ratio | ratio | p_weak_k / p_tls | add | Σ/Σ | BB | ratio |
| 31 | ja3_diversity | count | tls.ja3_ids | set | \|∪\| | NB | none |
| 32 | tls_handshake_ms | avg | p_hsms_s / p_hsms_n | add | Σ/Σ | t (log) | jensen |
| 33 | periodicity | window | derived.periodicity_score (slot grid, 6 h) | span | value at the span decision tick | t | — (H only, 6-h span ticks) |
| 34 | timing_regularity | window | derived.timing_regularity (slot grid, 6 h) | span | value at the tick | t | — (6 h) |
| 35 | think_time | avg | p_think_ls / p_think_n (n ≥ 2) | add | exp(Σ ln gap / n), the geometric mean | t (log = mean ln gap) | mean |
| 36 | req_per_session | window (log1p) | derived.req_per_session (24 h) | span | value at the tick | t | — (24 h) |
| 37 | duty_cycle | window | derived.activity_duty_cycle (slot grid, 24 h) | span | value at the tick | t | — (24 h) |
| 38 | retransmit_rate | ratio → t | p_retx_k / p_pkts | add | Σ/Σ. vec = logit((k+.5)/(n+1)) | t (logit) | mean |
| 39 | rtt | avg | p_rtt_s / p_rtt_n | add | Σ/Σ | t (log) | jensen |
| 40 | syn_ratio | ratio | p_syn / p_flows | add | Σ/Σ | BB | ratio |
| 41 | flow_duration | avg | p_dur_s / p_dur_n | add | Σ/Σ | t (log) | jensen |
| 42 | probe_reachable | gauge | p_reach_s / p_probes | add | probe-weighted mean | t | mean |
| 43 | probe_loss | ratio → t | p_loss_k / p_probes | add | Σ/Σ, logit | t (logit) | mean |
| 44–51 | comp_get, comp_write, comp_4xx, comp_5xx, comp_dns, comp_tls, comp_flows, comp_syn | clr | p_get, p_write, p_4xx, p_5xx, p_dns, p_tls, p_flows, p_syn | add | CLR of ln(Σc·60/cov + 0.5); nat = shares | t | mean |

Summary:
- 40 features are `add` (exact, and computable on rolling rows).
- 4 are `set` and 4 are `map` (exact up to HLL error above 4096 distinct,
  and up to per-tick top-64 truncation for maps; see §13.1 for tolerances).
- 4 are `span`.
- 40 features are Q-transfer-eligible, 8 are Q-native-only, and 4 are H-only.
- The 12 KEY_FEATURES of B14 are all `add` or `set`, so the H charts see
  exact values.

Gates, unchanged from FEATURE_SPEC but applied to the grain sums:
- A ratio with n = 0 is NaN.
- An average needs n ≥ 1 (think_time needs n ≥ 2).
- A bounded feature needs n ≥ 5.
- A count of an inactive window is a true 0. `feature.meta.<g>[active]` = 0
  means the row is not learned and not scored, as today with
  `feature.active`.

---

## 5. Data flow and store layout

### 5.1 Per-tick flow (canonical mode)

```
raw (R1 R2 R3)          + l4.peer_ids l4.dport_ids tls.ja3_ids act.template_ids act.slot_events
derived (D0 D1 D2)      slot-grid duty/periodicity/regularity, dns_dga_named_n, think_log_sum/gaps
B01  every tick         feature.{vec,nat,expo,active,tctx,sketch}  (tick rows, unchanged)
                        feature.part[47]; feature.live.h (+ .q if Δt<=900)
     H decision tick    feature.{nat,vec,meta,expo}.h, feature.sketch.h
     Q decision tick    feature.{nat,vec,meta,expo}.q
B03  every tick         H learners (clock feature.meta.h) and Q learner (clock feature.meta.q)
                        commit due grain rows; omega paired stats ride on the H current anchor
B04  H tick             behavior.z/zr/pf (H), scores marg_int marg_shape peer
     Q tick             behavior.z.q/zr.q/pf.q, scores marg_int_q marg_shape_q, behavior.prov
B05  H / Q tick         behavior.zi / zi.q, behavior.common.<grp> / common.q.<grp>, flags
B06  H / Q tick         t2 spe / t2_q spe_q, behavior.wh / wh.q
B07-B13 every tick      unchanged (slot / event / wall-clock clocks; T stream)
B14  H tick             cusum mcusum bocpd creep on zr (H), behavior.cusum_state
B15  H tick             windows = 4 committed active H rows -> model.identity fmt 2
B16/B17 H tick          attribution / linking on the last 4 active H rows
B18  H tick             class_int class_shape class_coherence (H rows); rhythm/novel per tick
B24  every tick         p for whatever was scored; strata by grain (H, Q) or cadence class (T)
B25  every tick         tick type tau; p_all -> q_all (meta ring dp|tau), e_day = q_all n_tau/beta_tau
                        S_t on T-stream inst detectors each tick; S_h on H-stream inst at H ticks
B26-B28 every tick      stream periods for e_day / acc levels
B29 / B30 / API         per-grain natural units and bands
```

### 5.2 New and changed series (canonical mode)

| Series | Owner | When | Layout | Retention |
|---|---|---|---|---|
| `l4.peer_ids`, `l4.dport_ids`, `tls.ja3_ids`, `act.template_ids` | R1, R2 | ticks with records | SetSketch dict | 75 min |
| `act.slot_events` | R2 | ticks with records | {slot_start: events} | 25 h |
| `derived.dns_dga_named_n` | D1 | with dns_dga_score | float | 2 h |
| `derived.think_log_sum`, `derived.think_gaps` | D2 | ≥ 1 selected gap | float | 2 h |
| `feature.part` | B01 | every tick | float32[47] (§3.1) | 2 h |
| `feature.live.h`, `feature.live.q` | B01 | every tick (q when observable) | float32[52] nat | 6 h |
| `feature.nat.h`, `feature.nat.q` | B01 | decision ticks | float32[52] nat | 8 d |
| `feature.vec.h`, `feature.vec.q` | B01 | decision ticks | float32[52] vec (dt := cov) | 2 d |
| `feature.meta.h`, `feature.meta.q` | B01 | decision ticks | float32[5] = [active, cov_s, n_ticks, n_active, G] | 8 d |
| `feature.expo.h`, `feature.expo.q` | B01 | decision ticks | dict {http, dns, tls, flows, probe: Σn} | 8 d |
| `feature.sketch.h` | B01 | H decision ticks | float32[80] | 2 d |
| `behavior.z`, `zr`, `zi`, `pf`, `wh` | B04–B06 | **H ticks** (H grain) | as v2 | **4 d** (96 H rows) |
| `behavior.z.q`, `zr.q`, `zi.q`, `pf.q`, `wh.q` | B04–B06 | Q ticks | as v2 | 6 h |
| `behavior.prov` | B04, B06 | Q ticks | {detector: π_nat} | 1 d |
| `behavior.score`, `pm`, `p` | emit, B24 | as v2 | float32[**35**] | 1 d |
| `behavior.common.<grp>`, `behavior.common.flag` | B05 | H ticks | as v2 | as v2 |
| `behavior.common.q.<grp>`, `behavior.common.flag.q` | B05 | Q ticks | as v2 | as v2 |
| `behavior.q_inst` (T stream), `behavior.q_inst.h` | B25 | every tick / H ticks | float32[1] | 8 d |
| `behavior.evidence` (S_t), `behavior.evidence.h` (S_h) | B25 | every tick / H ticks | float32[1] | 8 d |
| `behavior.cusum_state` | B14 | H ticks | as v2 | **4 d** |
| `behavior.class.agg` | B18 | H ticks | float32[52] | 9 d |

The tick rows (`feature.vec/nat/expo/active/tctx/sketch`) are unchanged.
They stay the representation for tick-native consumers: B07–B13, B02's cold
path, B09, and the API's `current_tick`. In tick mode the H names resolve to
the tick names (`grains.series`), so nothing new is read.

Retention is longest-prefix-wins. The `.h` and `.q` names inherit the
`feature.nat` / `feature.vec` rules. The 4-d rules for the H-grain behaviour
rings are raised by B04 and B14 with `ensure_retention` in canonical mode
only, so tick mode keeps today's memory.

### 5.3 Models (contract C amendments)

- **model.baseline@(s, e), fmt 2.** The top-level anchors are **H**:
  `current`, `reference`, `gate`, `gate_ref`, `golden`, `ref_elig`, `n_eff`,
  `held`, `allow_drift`, plus bookkeeping, all as in fmt 1. New fields:
  - `q: {current: Anchor(week=False, select=True), gate: GateState, n_eff, native_w}`;
  - `grain_mode`.

  The H current `Anchor` gains `om[4, 52]`: decayed paired-hour sums A, B,
  D, W (§6.3). They are folded with the row, so checkpoint, rollback,
  release and rebase cover them.

  `n_eff` is now in **15-min equivalents**, Σ w·cov/900: an H row counts 4,
  a Q row 1. The B01 `stable` (≥ 96), B24 maturity (≥ 48) and B03
  `GOLDEN_MIN_NEFF` (≥ 96) thresholds keep their meaning.

  A fmt-1 model loads as an H-only fmt 2 (anchors relabelled H, Q empty,
  `om` zero) in tick mode. In canonical mode it is reset, because its rows
  were tick rows.
- **Tier models** `model.baseline@(s, class:<rid> | __system__)` and
  `@(__org__, __org__)` keep being H. New field `omega {A, B, D, W}[52]`,
  the sums of the members' `om`, plus `omega_eff {omega, delta}[52]` after
  the EB shrink (§6.3).
- **model.density** stays H. **model.density.q** is new: a Q fit shrunk to
  the entity's H density with 30 pseudo-rows. **model.groups** is fitted
  from H zi only.
- **model.cp** state is per H tick. **model.classagg** holds H rows.
- **model.identity fmt 2** fits windows of 4 H rows. A fmt-1 model reads as
  unfitted. **model.idwin** buffers are keyed by H rows.
- **model.calib** ring keys:
  - H and Q detectors: `'<d>@<daypart>|g:h'` and `'<d>@<daypart>|g:q|p:<0|1>'`;
  - identity: `'identity@<daypart>|r<k>|g:h'`;
  - T-stream detectors keep `'<d>@<daypart>|<cc>'`;
  - B25 meta rings: `'meta_inst_h@<daypart>'`, `'meta_inst_t@<daypart>|<cc>'`
    and `'meta_all@<daypart>|t:<τ>'` (with `'|<cc>'` appended for τ = t).

---

## 6. Multi-resolution baselines with learned scale transfer (B03, B04, B06)

### 6.1 Learners (B03)

| Learner (checkpoint name) | Clock | Rows | Window trust | Delay D |
|---|---|---|---|---|
| `baseline.current` (H) | `feature.meta.h` | H decision rows with active = 1 | w_eff = min trust over (ts − 3600, ts]; w_prov = min trust_prov | max(4Δt, 600 s) |
| `baseline.reference` (H) | `feature.meta.h` | as v2, 24-h delay, trust = 1, no incident or regime ±24 h | as above | 24 h |
| `baseline.current.q` (Q, new) | `feature.meta.q` | Q decision rows with active = 1 | min over (ts − 900, ts] | max(4Δt, 600 s) |

- **Commit rule.** Rows are held while quarantine(now − Δt) = 1, as in v2.
- **Row exposure.** The row is `make_row(ts, nat, dt = cov, tctx = row_tctx(ts))`.
  This replaces `_row_dt`'s gap heuristic, because cov is the true exposure.
- **Why min trust.** An hour that contains any untrusted minute is not a
  clean hour.
- **Rate caps.** The H current cap is 0.1 σ15 per band-day and the
  reference cap 0.03 σ15, exactly as in v2. σ15 is taken from the Q-grain
  predictive obtained by transfer (§6.2) of the bucket's H posterior, so
  "0.1 σ15" keeps its architecture meaning.
- **Q anchor.** The Q anchor has the same cap, no reference, no golden
  snapshot and no bin168 cells. Its location refinement comes through the
  transfer parent.
- **Tiers.** The tier refresh is unchanged and uses the H anchors. It adds
  the `omega` sums and the EB-shrunk `omega_eff`.

### 6.2 Exact predictives per grain (B04 through `m_baseline.predictive_set(..., grain=g)`)

For each grain the bucket is b = bin48 of `row_tctx(now, g)`. The exposure
is e_g = cov/60 minutes (count families). The n of a ratio is the grain
Σn. The value y is the grain vec.

**H (primary).** The families are those of v2, fitted on H rows only, so the
exposure is ≈ 60 min for every row:

- NB: X_h ~ NB(mean μ_H·e_h, size r_H), r_H = 1/(1/κ̂_H + 1/a_H). κ̂_H is
  the moment overdispersion of the H stats, clipped to [0.5, 1e3]. a_H is
  the posterior Gamma shape.
- BB: k_h ~ BB(n_h, p_H c_H, (1 − p_H) c_H), c_H = min(Σn + 2, φ̂_H),
  φ̂_H ∈ [20, 1000].
- t: y_h ~ t_{2α}(m, β(κ + 1)/(ακ)) from the NIG of H rows, weight
  min(1, n/5).
- Dual anchor: p_f = min(1, 2 min(p_cur, p_ref)).
- Class tier: `peer` uses the leave-one-out H tier, as in v2.

**Q transfer.** Take the H predictive at b (current, or reference/golden),
the entity's chain values v_f and δ_f (§6.3), and the family. The
transferred predictive T is then:

- NB (`rate`): μ_T = μ_H (per minute), r_T = r_H / v_f.
- BB (`ratio`): p_T = p_H, c_T = max(C_T_MIN, (1 + c_H)/v_f − 1).
- t (`jensen`, `mean`): loc_T = loc_H + δ_f, scale_T = scale_H·√v_f,
  df_T = df_H.
  - Default δ_f for `jensen`: −(v_f − 1)·σ²_H/2, with σ²_H = scale_H²·df/(df − 2)
    (scale_H² if df ≤ 2). This is the lognormal identity: equal
    expectations of the rate at both grains imply μ_Q = μ_H − (σ²_Q − σ²_H)/2.
  - Default δ_f for `mean`: 0.
  - The learned δ̂ replaces the default through the EB chain (§6.3). |δ|
    is clipped to ≤ scale_T.

**Q predictive (current).** E_Q(b, f) = S_Q,native(b, f) ⊕ Pseudo(T_cur, κ_T).
- S_Q,native is the entity's Q current anchor stats at b.
- Pseudo(·, κ) is κ = 16 rows at exposure 15 min (count), n̄_Q = n̄_H/m
  trials (ratio), or unit weight (t). Their sufficient statistics match
  the moments of T exactly:
  - NB: Σe = 15κ, Σx = 15μκ, Σx² = κ[(15μ)² + 15μ + (15μ)²/r_T], and the
    cross moments Σxe and Σe² from the constant exposure.
  - BB: Σk, Σn and Σk²/n chosen so that φ̂ = c_T.
  - t: Σy = κ·loc_T, Σy² = κ(loc_T² + σ²_T).

  The native rows therefore dominate once W_native ≫ κ_T. This is the
  conjugate "prior as pseudo-data" rule that `m_baseline.join` already uses
  for tiers.
- For `none` features the Q predictive is S_Q,native with the hyperprior.
  The feature is **scored only when W_native(b, f) ≥ κ_T**, otherwise its
  Q p is NaN.
- **Provenance:** π_nat(b, f) = W_native/(W_native + κ_T). A Q score is
  **provisional** when the median π_nat over the features scored on that tick
  is < 0.5. Then B04 and B06 write `behavior.prov[d] = π_nat`.

**Q predictive (reference).** This is T_ref, the transfer of the H reference
(golden once it exists). It has no native part. So p_f,Q = min(1, 2 min(p_Qcur,
p_Qref)) keeps the dual-anchor creep protection at no extra state.

**Tick mode.** `predictive_set(..., grain='h')` is v2's `predictive_set`.

`quantiles(pred, qs, dt_s = G_g)` and `mean_nat(pred, dt_s = G_g)` give
natural units per grain: counts per hour and per 15 min, ratios as
fractions, averages in their unit.

### 6.3 The variance-scale factor ω_f (paired hours)

**Paired hour.** A committed H row (cov ≥ 0.95·3600) that contains exactly
m = 4 Q decision rows, each with cov ≥ 0.95·900 and a finite value of f.
This only happens at Δt ≤ 900. The H row's commit weight w applies. With the
Q values w_1..w_4 on the family's working scale, s² their unbiased sample
variance, and the H predictive taken at the row's bucket *before* the row is
folded:

| Family | Working value w_j | a (excess within-hour variance) | b (H-grain excess variance) | d (location) |
|---|---|---|---|---|
| NB | x_j/e_j (rate per min) | s² − λ̂/ē_Q, with λ̂ = X_H/e_H | λ̂²/r_H | — |
| BB | k_j/n_j (all n_j > 0) | s² − p̂(1 − p̂)·mean(1/n_j), with p̂ = K/N | p̂(1 − p̂)/(1 + c_H) | — |
| t | y_j (Q vec) | s² | σ²_H (predictive variance of y) | mean(w_j) − y_H |

The H anchor folds A += w·a, B += w·b, D += w·d, W += w (decayed with the
anchor's half-life). The tier sums are Σ over members.

**EB chain.** For the entity's chain entity → class (≥ 3 members) → system
→ org:

```
omega_default = M                                  (v = M: independent quarters)
omega_node    = (A + K_OM * omega_parent * bbar) / (B + K_OM * bbar),  bbar = B/W  (W = 0: omega_parent)
delta_node    = (D + K_OM * delta_parent) / (W + K_OM)                (delta_parent at org: the family default)
omega         = clip(omega_node, 0, 2M);   v = 1 + ((M-1)/M) * omega
```

Here K_OM = KAPPA_OMEGA = 24 pseudo paired-hours.

**Why this is exact.** Suppose the H working value is the mean of the m
quarter values, and the quarters are exchangeable within the hour. Then
Cov(h, q_j − h) = 0, so Var(q) = Var(h) + ((m − 1)/m)·E[s²]. For NB the
Poisson parts cancel exactly (1/e_Q − 1/e_H = (m − 1)/(m e_Q)), which leaves
1/r_Q = (1/r_H)·(1 + ((m − 1)/m)ω). Independent quarters give ω = m and so
v = m. Fully correlated quarters give ω = 0 and v = 1.

A gamma-mixed Poisson simulation with 2·10⁵ hours (λ = 2/min, log-sd 0.6)
reproduced the true Q overdispersion to 4 digits at intra-hour correlation
0, 0.5, 0.9 and 1. The script lives in the session scratchpad and becomes
`tests/lib/test_grains.py::test_omega_recovers_quarter_overdispersion`. For
BB (unequal n_j) and log-t (Jensen), the decomposition is approximate. There
δ̂ carries the location part, and the (daypart, q, prov) calibration rings
absorb the rest (§9.1).

**Default when there are no paired hours** (warm-up at 3600, pack D, mini):
ω = m, i.e. v = 4. For NB this multiplies **only the overdispersion term**
(Var_Q = μ_Q + v·μ_Q²/r_H). For pure-Poisson features the default is
therefore exact. It is the widest Q predictive that non-negative within-hour
correlation allows, so it is conservative in the sense the lead asked for.

### 6.4 B06 per grain

H: T² and SPE on H zi at H ticks. The density is refitted every 16 H rows
or 4 h on ≤ 336 committed H rows (14 d).

Q: T²_q and SPE_q on Q zi at Q ticks, from `model.density.q`:
Σ̃_Q = (n Σ_Q + 30 Σ_H)/(n + 30), the same shrink B06 already applies to a
young entity toward its class. The result is provisional while n_Q < 64.

`model.groups` (dependence groups) is fitted from H zi only and used by B04
at both grains.

---

## 7. Overlap-aware sequential detection and budgets

### 7.1 What is sampled where

| Consumer | Input | Updates | Overlap between consecutive inputs |
|---|---|---|---|
| B03 H / Q learners | decision rows | 1 per G | none |
| B04, B05, B06 H | H decision rows | 1 per hour | none |
| B04, B05, B06 Q | Q decision rows | 1 per 15 min | none within Q. Q is nested in H (handled in §7.3) |
| Span features | span decision ticks | 1 per 6 h or 24 h | none |
| B14 cusum, mcusum, bocpd, creep | H zr | 1 per hour (creep daily) | none |
| B16 identity (instantaneous score) | last 4 active H rows | per active H row | 3 of 4 rows. **Marked `overlap`**: single-tick path only, never in S_h |
| B16 identity CUSUMs | non-overlapping 4-row windows | every 4th active H row | none |
| B18 class_int, class_shape, class_coherence | members' H rows | 1 per hour | none |
| T-stream detectors | tick contents | every tick | none (disjoint ticks) |

### 7.2 Charts (B14) and identity CUSUMs (B16)

- **ARLs in grain periods.** `seq.h_for(kind, arl_days, period_s)` with
  period = G_h, so thresholds do not depend on Δt:
  - cusum (2400 d, k = 0.25): h = 16.60 at every cadence. In v2 it was
    19.37 at 900 s and 24.79 at 60 s.
  - mcusum (d = 12, 100 d): h = 22.72 at every cadence (v2 at 900 s: 25.50).
- **AR(1) φ** is re-learned on H lag-1 pairs. A pair is adjacent when the
  two H rows are G ± Δt apart. `PHI_MIN_PAIRS` stays 48, and `phi_at(dt)`
  becomes a single φ.
- **BOCPD** takes the H zr intensity mean and `wh` directly, so its internal
  hour roll-up goes away. Creep uses daily means of H rows.
- **B16** steps `cusum_other` and `cusum_new` only on windows that share no
  row with the previous step. h = `seq.h_for('llr', 100 d, …)` with windows
  per day = (active H rows per day)/4, which is v2's rule with H rows in
  place of ticks.

### 7.3 Streams, tick types and the single-tick budget (B25)

The registry gains `DETECTOR_INFO[d]['stream']` ∈ {h, q, t} and
`['overlap']` (identity only):

- **h:** marg_int, marg_shape, peer, t2, spe, cusum, mcusum, bocpd, creep,
  identity, class_int, class_shape, class_coherence.
- **q (appended; D = 35):** marg_int_q, marg_shape_q, t2_q, spe_q. Families
  are intensity, shape, intensity and shape. There is no `peer_q`, because
  there is no class Q tier.
- **t:** offhours, silence, novelty, novelty_rate, jsd, client, seq, dwell,
  timing, beacon, budget_vol, budget_exfil, budget_breadth, class_rhythm,
  class_novel, mixture, session, cross_system.

On every tick, B25 combines whatever was scored:
p_all = wHMP(all families), with NaN meaning not scored (a non-decision tick
is not degraded). It then meta-calibrates in the stratum (daypart, τ),
adding cc when τ = t. The meta ring is conformal within that stratum, so
q_all is uniform under the null on ticks of type τ, and

```
e_day = q_all * n_tau / beta_tau        (beta renormalised over tick types with n_tau > 0)
```

The expected number of null single-tick alarms is Σ_τ n_τ·(0.03 β_τ/n_τ) =
0.03 per entity-day at every Δt. HMP is robust to the dependence between
nested H and Q evidence on the same tick.

| Δt | Types present | Multiplier n_τ/β_τ | q for e_day ≤ 0.03 | v2 (every tick) |
|---|---|---|---|---|
| 3600 | h | h: 24 | 1.25e-3 | 1.25e-3 (identical) |
| 900 | h, q | h: 36; q: 216 | 8.3e-4; 1.4e-4 | 3.1e-4 |
| 300 | h, q, t | h: 48; q: 288; t: 768 | 6.3e-4; 1.0e-4; 3.9e-5 | 1.0e-4 |
| 60 | h, q, t | h: 48; q: 288; t: 5376 | 6.3e-4; 1.0e-4; 5.6e-6 | 2.1e-5 |

Hourly evidence costs about the same at every cadence (1.25e-3 … 6.3e-4).
In v2 it cost 60× more at Δt = 60.

Severity thresholds (e_day ≤ 0.03, 3e-3, 3e-4, 3e-6) are unchanged. They
apply to this e_day.

- **Accumulator e_day:** e_acc = min_d p_d · 86400/period_s(d, Δt).
- **Family e_day** (corroboration, reading rules):
  p_family · 86400/(min period among the members scored on this tick). This
  is conservative.

### 7.4 Evidence CUSUMs per stream

Budget 0.03 per entity-day, split by EVIDENCE_SHARE:

| Chart | Input | Updates | ARL | h |
|---|---|---|---|---|
| S_t (`behavior.evidence`) | q_inst,t = meta-calibrated wHMP of the T-stream instantaneous detectors (novelty, client, seq, dwell, P2 session / cross_system) in the ring `meta_inst_t@dp\|cc` | every tick | 66 d in ticks | 4.57 (3600), 6.05 (900), 7.22 (300), 8.93 (60) |
| S_h (`behavior.evidence.h`) | q_inst,h = the same over the H-stream instantaneous detectors **except identity** (overlap): marg_int, marg_shape, peer, t2, spe, class_int, class_shape, class_coherence, in `meta_inst_h@dp` | H ticks | 66 d in hours | 4.57 at every cadence |

The Q stream feeds no CUSUM. Its windows are nested in the H window of the
same hour, so they would count the same data twice. The Q stream is the
sensitivity path: single-tick evidence at 15-min resolution. Moderate
persistent shifts accumulate through S_h and B14.

- **Update rule:** S = max(0, S − ln q − 3), as in v2. An evidence-only
  alarm is LOW, or MEDIUM when S ≥ 2h.
- **Audit** (block bootstrap) runs per stream. Blocks are 1 h of T ticks or
  24 H rows.
- **trust_prov** uses e_inst = min_s q_inst,s·N_s, where N_t = 86400/Δt and
  N_h = 24.
- **Corroboration windows are wall-clock per stream:**
  - "within 4 ticks" becomes within max(4Δt, 1 h) for H evidence and
    max(4Δt, 15 min) for Q;
  - "2 consecutive ticks at ≤ 3e-3" becomes 2 consecutive decision ticks of
    the same stream.

### 7.5 Provisional Q evidence

- A Q detector with `behavior.prov[d]` < 0.5 is calibrated in its own
  B24 stratum (daypart, q, prov = 1).
- It enters its family with weight × 0.5.
- It can raise an alarm up to MEDIUM. It never counts toward HIGH or
  CRITICAL corroboration.
- B28 excludes it from SUSPECT triggers. As native support grows, the
  prov = 0 stratum takes over.

---

## 8. Identity windows (B15, B16, B17, B24)

- **Row:** `m_identity.grain_row(store, s, e, ts)` = [feature.vec.h 52 |
  feature.sketch.h 80 | behavior.timing (B, M, think_mu) at ts 3 | clock
  features of `row_tctx(ts, 'h')` 3] = 138 values. This is `tick_row`'s
  layout, with the grain row in place of the tick row. In tick mode it *is*
  `tick_row`.
- **Window:** 4 active H rows. `K_WIN = 4` keeps its value, and its unit is
  H rows. That is 4 h of wall-clock data at every cadence. At 3600 s the
  window is identical to v2's window, so B15's warm-up enrolment is
  unchanged.
- **Full window** (B16's unknown CUSUM rule): 4 active H rows, each with
  cov ≥ 0.95·3600, within a 48-h lookback. The lookback was 24 h and was
  widened for sparse entities, so `feature.vec.h` and `feature.sketch.h` are
  retained 2 d.
- **B15 collection:** non-overlapping windows of 4 committed active H rows,
  clock `feature.meta.h`. The fit runs every 24 h of wall clock;
  `fit_ticks` is ignored in canonical mode. `CAL_GAP_S` (4 h) is unchanged.
  `model.identity` is fmt 2.
- **Modal data** (vocab, client, rhythm, seq, timing) comes from
  `grain_modal_data(store, s, e, ts)`, which merges act.tokens,
  client.stack_set and act.stream over the H window at the H tick. B16
  caches it per H row, as it caches tick data today.
  - The cheap LLRs (vocab, client, rhythm) are additive over counts, so they
    are computed once per H row from the merged counts instead of per tick.
    At Δt = 60 that is 60× fewer evaluations.
  - The slow modalities (PPM, gap histogram) are computed at window
    completion.
- **B24 identity stratum:** (daypart, regime tercile, **grain h**) replaces
  (daypart, tercile, cc). The 3600-s warm-up rings are then the right rings
  for live 900-s and 60-s attribution. The integration fix "per cadence
  class" becomes unnecessary and is superseded. The ring content is
  comparable because the windows are.

---

## 9. Downstream consistency

### 9.1 B24 calibration

- **Stratum per detector:**
  - H stream: (daypart, 'h');
  - Q stream: (daypart, 'q', prov);
  - identity: (daypart, tercile, 'h');
  - T stream: (daypart, cc), unchanged.

  The daypart comes from `row_tctx` for H and Q, and from the tick for T.
- H rings fill at 24 per day at every cadence. They are **not** reset or
  thinned by a cadence switch. The 64-entry threshold and the pm and class
  blends are unchanged.
- **Admission** is at the detector's decision ticks only: its score is NaN
  on other ticks and skipped. Maturity is n_eff in 15-min equivalents.
- **Health:** the realised rate of e_day ≤ 0.03 uses p·86400/period_s(d).
  The KS test is per detector, pooled over strata, as in v2.

### 9.2 B25 fusion

§7.3–§7.5. The meta rings are keyed by (daypart, τ[, cc]).

- **Pending** stores (stratum, τ, Δt) so that B29 can recompute.
- **BH** across a system's keys applies to LOW single-tick alarms, per tick,
  as in v2.
- The degraded-family logic is unchanged, and non-decision NaNs are not
  degradation.
- `behavior.alarm` gains `stream` for evidence-path alarms.

### 9.3 B26, B27, B28

- **B26:** `excess_surprise(p, period_s)`. Family terms use the family
  period from §7.3.
- **B27:**
  - `acc_level_from_p(p, d, period_s(d, Δt))`, so detectors.acc_level uses
    ARL in grain periods for H accumulators;
  - the quiet close needs both q_inst,t and the latest q_inst,h (≤ 1 h old)
    to be quiet;
  - the minimum close time max(8 ticks, 2 h) is unchanged.
- **B28:**
  - `evidence_factor` takes e_inst (§7.4);
  - `_ln_arl` works per detector period;
  - the level and ramp series are zr (H) at H ticks, with `feature.vec.h`
    as fallback, held between H ticks. The Sen slope bins are hourly.

### 9.4 B29 explain

- **Numeric attribution** happens at the grain(s) that scored the trigger
  tick. H is used when `marg_*`, t2 or spe drove the incident, Q when a
  `_q` detector did. Natural units are per hour or per 15 min, from the
  per-grain `model_state` (§9.5).
- **Neutralising** a feature resets its grain value to the bucket median of
  that grain's predictive.
- **Replay** of cusum and mcusum iterates the H decision ticks (the
  `behavior.cusum_state` ring is 4 d = 96 H states). This covers the "≤ 96
  ticks" window at every cadence; v2 held only 6 h.
- The evidence CUSUM is replayed per stream. e_day uses the tick type
  stored in B25's pending entry. `m_calib.p_replay` takes grain and prov.
- **Scope** lists the grains recomputed.

### 9.5 B04 model_state, B30 portrait, API

- **`profile.extra.model_state`** becomes `{grains: {h: {…}, q: {…, prov}}}`
  per the v2 layout, at `dt_s` = 3600 and 900. The legacy top-level fields
  are those of Q when Q is observable, else H. It is refreshed hourly per
  grain at a per-entity phase.
- **B30:**
  - Workload bands p5/p50/p95 are given per grain and day type: "每小时 /
    per hour" from the H anchor, "每刻 / per 15 min" from the Q predictive
    with a `provisional` flag.
  - The legacy per-tick band is filled with the Q band, with exposure_s 900.
  - Class bands come from B18's H agg anchors.
  - The zh and en templates name the grain.
- **API** (`views.feature_rows`, `routes_v2._current_workload`):
  - Each feature row gains `grains: {h: {current, p5, p50, p95, z, zr},
    q: {current, p5, p50, p95, z, zr, provisional}}`.
  - `current` comes from `feature.live.<g>`, which is exact for distinct
    counts. v2 scaled the tick value by 900/Δt, which is wrong for
    sub-additive features.
  - The legacy fields (`current`, `p5`, `p50`, `p95`, `z`, `zr`, `baseline`,
    `spread`, `stable`, `current_tick`) stay. They are filled from Q when Q
    is observable, else from H, and `unit` names the grain.
  - `tests/api` assertions (field presence, no NaN) keep holding.

### 9.6 Tick-native detectors (what remains cadence-dependent)

- B07 (15-min slot clock), B11, B12 (stream timestamps) and B13 (wall-clock
  horizons over tick sums) are cadence-invariant by construction and do
  not change.
- Some statistics depend on the tick's sample size: B08 novelty, jsd and
  novelty_rate, B09 client, and B10 seq and dwell. They keep cadence-class
  strata in B24.
- Their rings are empty after a cadence switch. This is why D12 requires
  the final warm-up phase at the **live** cadence.
- Moving jsd and novelty_rate onto H-grain token distributions is a
  follow-up (§16).

### 9.7 Eval harness

- **runner:**
  - `_baseline_snapshot` is per grain.
  - The feature collector also reads `feature.nat.h` and `feature.meta.h`
    at H ticks.
  - The stale checker already uses each series' median write interval as
    its period, so hourly grain series are not flagged. A test pins this.
- **metrics:**
  - Gate 7 exceedance is computed per (cadence class × tick type).
  - Gate 8 adds a "per H row top-1" figure next to the per-tick one.
  - Gate 11 coverage is computed per grain: held-out H rows against H bands,
    and held-out Q rows against Q bands.

---

## 10. Packs and runtime (a pack-definition fix, not seed tuning)

**Rule (packs.py `warmup_phases(tl_start_date, calendar, tz, live_dt, total_days=16)`, new):**
- If live_dt ≤ 900, the warm-up is [((16 − k)·24, 3600, agg), (k·86400/live_dt,
  live_dt, live_dt ≥ 900)]. k ≥ 2 is the smallest number of local days
  before the scenario start whose span contains ≥ 1 full workday and ≥ 1
  full non-workday under the pack's calendar (holidays and make-up days
  included).
- If live_dt = 3600, the warm-up is 16 d at 3600.

Why ≥ 1 day of each type: κ_T = 16 weighted rows is about one day of a
bucket at Δt ≤ 900. v2's von Mises spread gives ≈ 3.7 weighted rows per
row-hour per bucket, so one day at 4 Q rows per hour gives ≈ 15. After one
day of each type, every bin48 bucket has π_nat ≈ 0.5. The B24 cc-900 rings
reach 64 entries in about 1.3 workdays.

| Pack | Scenario start | v2 warm-up | v2.1 warm-up | 900-s days |
|---|---|---|---|---|
| A | Mon 2025-03-10 | 336 × 3600 + 192 × 900 (Sat, Sun) | 312 × 3600 + 288 × 900 | Fri, Sat, Sun |
| B | Thu 2025-06-12 | same (Tue, Wed) | 288 × 3600 + 384 × 900 | Sun, Mon, Tue, Wed |
| C | Fri 2025-03-28 (Berlin) | same (Wed, Thu) | 264 × 3600 + 480 × 900 | Sun … Thu |
| D | Thu 2025-09-04, live 3600 | same | 384 × 3600 | — (Q not observable live) |
| E | Tue 2025-03-11, live 60 | 672 × 900 | unchanged (7 d at 900 covers both types) | Q native; cc-60 rings start empty by design |
| smoke | Mon 2025-03-10 | 120 × 900 (Sat 18:00 →) | 96 × 3600 + 24 × 900 (Wed 18:00 → Sun 24:00): 120 ticks, so the CPU gate is unchanged | Sun evening (Q provisional on Monday: the transfer path) |
| mini | Mon 2025-03-10 | 72 × 3600 | unchanged (Fri, Sat, Sun at H) | none: CI coverage of the transfer-only path |

- The seeds, scenarios and onsets do not change. Scenario-phase tick
  indices are unchanged because the scenario phase is unchanged.
- The extra 900-s warm-up ticks cost about +96 (A) to +288 (C) ticks per
  pack-seed (§15).
- **Runtime** (`build.Runtime`): the warm-up plan becomes
  `[(120, 3600), (192, 900)]` (5 d + 2 d) instead of 180 × 900. It is
  configurable, and it warns when the 900-s phase does not cover both day
  types. Live stays at `window_s` (60). The Q-grain is native from the
  900-s phase. `scripts/smoke.py` prints the plan.

---

## 11. Per-engine change list

The "Tests" column names new tests (new files are marked). Existing tests
stay green in tick mode. Existing expectations change in exactly two places,
both deliberate spec changes:
1. `tests/lib/test_lib_data.py::test_detector_registry` (N_DETECTORS 31 → 35, last
   detector `spe_q`) in M0.
2. The v2-tick-semantics assertions of `tests/test_pipeline_e2e.py` when the
   default flips to canonical in M8 (§12).

| File | Function(s) | Change | Tests |
|---|---|---|---|
| `lib/grains.py` (new) | all (helpers_api "grains") | Constants, observable / decision / span decision, tick type, n_per_day, beta, e_day_tick, e_day_detector, e_inst, period_s, evidence_arl, row_tctx, series (tick-mode name resolution), last_decision, scored_mask, transfer_nb / bb / t, v_from_omega, omega_chain maths | `tests/lib/test_grains.py` (new): decision at 60/300/900/3600, misaligned, switches; Σ n_τ = 86400/Δt; the tick-mode e_day equals `combine.e_day`; β renormalisation; transfer moments; ω simulation (§6.3) |
| `lib/features.py` | new PART_SPEC, PART_NAMES, PART_DIM, GRAIN_CLASS, TRANSFER, SPAN_FEATURES, `compute_parts(get, dt)`, `grain_values(S, cov, G, sets, maps, span)` | §3–§4 table as data | `tests/lib/test_features_grains.py` (new): grain_values(Σ parts) equals compute_features at Δt = G for one tick; gates; NaN rules |
| `lib/sketch.py` | `set_sketch`, `SetSketch.union`, `count` | §3.2 | `tests/lib/test_sketch.py` (+): exact ≤ 256, exact union ≤ 4096, HLL beyond, union associativity, and union of the sketches equals the sketch of the union |
| `lib/detectors.py` | DETECTORS (+4 appended), `_TABLE`, DETECTOR_INFO `stream`, `overlap`, `strata`; `acc_level(p, d, dt)` uses `grains.period_s` | §7.3 | `tests/lib/test_lib_data.py`: N_DETECTORS 31 → 35 and the last element (deliberate) |
| `lib/gating.py` | `commit_candidates(..., window_s=None)`, `GatedLearner.window_s` | trust = min over (ts − window_s, ts] | `tests/lib/test_gating.py` (+): window-min trust; the default None keeps v2 |
| `lib/calib.py`, `lib/m_calib.py` | `grain_stratum_key`, `meta_stratum_key`; `stratum_for(..., grain=None, prov=0)`, `ring_key_for`, `p_replay`, `issued_stratum` | §5.3 keys | `tests/lib/test_calib.py` (+) |
| `lib/seq.py` | none (callers pass period_s) | — | — |
| `core/store.py` | DEFAULT_RETENTION (+ feature.part 2 h, feature.live 6 h, feature.meta 8 d, feature.sketch.h 2 d, raw `*_ids` 75 min, act.slot_events 25 h, behavior.prov 1 d, behavior.*.q 6 h) | §5.2 | `tests/core/test_store_v2.py` (+) |
| `core/engine.py` | `default_config` | `grain_mode` ('tick' until M8) | — |
| `raw/l4flow.py` | `run` emit block | `l4.peer_ids`, `l4.dport_ids` from the full `a.peers` / `a.dports` | `test_r1_raw.py` (+) |
| `raw/tls.py` | `run` | `tls.ja3_ids` | `test_r1_raw.py` (+) |
| `raw/action_token.py` | `_emit` (+ `_Acc` slot tally) | `act.template_ids`; `act.slot_events` (aggregated records split over ts_sample) | `test_r2_action_token.py` (+): Σ slot events = act.events; the aggregated split |
| `derived/entropy.py` | `run` | `derived.dns_dga_named_n` | `test_d1_instant.py` (+) |
| `derived/session.py` | duty computation, think gaps | canonical: duty on the 15-min slot grid (covered-slot denominator), with tick-grid fallback without `act.slot_events`; `think_log_sum` / `think_gaps` in both modes | `test_d2_session.py` unchanged (tick mode and fallback keep v2); new: equal duty at 60/900/3600 for the same events (canonical) |
| `derived/periodicity.py` | `_entity`, `regular_counts` | canonical: per-slot events grid (24 bins / 6 h), MIN_TICKS on slots; tick mode and no-slot fallback unchanged | `test_d0_window.py` unchanged (`test_periodicity_undefined_…` writes no slot events, so it keeps its v2 expectation); new: defined at 3600 and equal at 900/3600 from slot events (canonical) |
| `derived/aggregation.py` | `ensure_retention` | + act.slot_events | — |
| `behavior/feature_vector.py` (B01) | `run`, `_entity`, new `_grain_rows`, `_window`, `_union_sets`, `_merge_maps` | parts ring; live rows; decision rows; sketch.h; raise raw-set retention | `tests/engines/test_b01_grains.py` (new): additive exactness at Δt = 60 over 60 ticks (1e-6); set unions; cov gate; decision-only writes; live every tick; no Q at 3600; midpoint tctx; tick mode writes nothing new |
| `behavior/lib/m_baseline.py` | `Row` (+grain, pair), `make_row`, `Anchor` (+om), `_fold48` / `commit_many` (+paired stats), `new_model` / `load` (fmt 2, migration), `predictive_set` / `predictive` / `quantiles` / `mean_nat` / `own_support` / `n_eff` (+grain), new `omega_chain`, `transfer_pred`, `pseudo_stats`; σ15 from the Q transfer | §6 | `tests/engines/test_b03_grains.py` (new): pseudo-stat moments give T exactly; native dominance after 4κ_T; π_nat; `none` features unscored before κ_T; ω fold / rollback exactness (checkpoint + replay == offline within 1e-6) |
| `behavior/baseline.py` (B03) | `__init__` (Q learner), `_learn`, `_fetch_cur` / `_fetch_ref` (grain rows, cov, row_tctx, paired Q rows), delete `_row_dt`, `_publish` (q block, n_eff units), `_refresh_tiers` (omega), `_profiles` | §6.1 | `test_b03_grains.py`: learners commit decision rows only; window-min trust; an H row with an untrusted minute has w = 0; tier omega sums |
| `behavior/likelihood.py` (B04) | `run` (per due grain), `_entity(…, grain)`, `score_features(dt = cov)`, span mask, `_model_state` per grain, `_degraded` per grain; DETS_Q | §6.2, §7.1 | `tests/engines/test_b04_grains.py` (new): Q NB r = r_H/v, BB c, t loc/scale; `none` NaN; prov; null KS of Q-transfer p on simulated quarters is super-uniform (never anti-conservative at v = m) |
| `behavior/common_mode.py` (B05) | `run`, `_system`, `_learn`, `_aggregates` (names per grain), learner 'common_mode.q' | per grain | `test_b05_common_mode.py` (+): the H pass only at H ticks; Q names |
| `behavior/multivariate.py` (B06) | `run`, `_entity`, `_score`, `_learn`, `_refit`, `_system_models`; `_fit_groups` (H only); density.q shrink; prov | §6.4 | `test_b06_multivariate.py` (+): Q shrink to H; the H refit uses hourly rows |
| `behavior/changepoint.py` (B14) | `run`, `_entity` (H ticks), `_learn_update` / `_refit` (H lag-1 φ), `phi_at`, `_roll_hour` / `_close_hour` (removed in canonical), `_outputs` (h from period), `_idle_outputs`, `_creep` (daily H means) | §7.2 | `tests/engines/test_b14_grains.py` (new): an N(0,1) null of hourly rows at Δt = 60/900/3600 gives the same alarm rate per day (±50 %, 200 d); the chart does not move between H ticks |
| `behavior/lib/m_identity.py` | `grain_row`, `grain_modal_data`, `window_from_store(…, grain)`, FMT 2 | §8 | `tests/engines/test_b15_grains.py` (new): identical traffic re-batched at 60/900/3600 gives the same window vectors (tolerance §13.1) |
| `behavior/identity_model.py` (B15) | `_fetch`, `_collect`, `_calibrate_live`, fit period | §8 | as above |
| `behavior/attribution.py` (B16) | `_system`, `_entity` (H ticks, 4 active H rows, 48 h), `_cheap` (per H row), `_slow`, `_mismatch` / `_unknown` (non-overlapping steps), LOOKBACK_S | §8, §7.2 | `tests/engines/test_b16_grains.py` (new): an enrolled entity at 3600 s is attributed self at 900 s and 60 s with no unknown_identity over a clean day |
| `behavior/entity_link.py` (B17) | `_active_ts`, `_z_now`, `_evaluate`, `_score_window` | H rows | `test_b17_entity_link.py` (+ canonical case) |
| `behavior/class_monitor.py` (B18) | `_class` (H ticks for int / shape / coherence), `_fetch_cur` / `_fetch_ref` (H rows, cov), `_int_shape` | H rows | `test_b18_class_monitor.py` (+) |
| `behavior/calibration.py` (B24) | `_fetch`, `_update`, `_score`, `_prior`, `_keys_for`, `health_add` (period) | §9.1 | `tests/engines/test_b24_grains.py` (new): H rings continue across a 3600 → 900 → 60 switch; Q prov strata; admission only at decision ticks |
| `behavior/fusion.py` (B25) | `_phase1` (τ, meta keys, e_day), `evidence_h` / `evidence_update` (per stream), `_finalise` (corroboration windows, prov weights), `_audit` (per stream), `fuse` (prov weight) | §7.3–§7.5 | `tests/engines/test_b25_grains.py` (new): null uniform p streams at 60/900/3600 give single-tick alarms 0.03 ± 50 % per entity-day and evidence alarms ≤ 0.045 per entity-day (200 d); the tick mode equals v2 |
| `behavior/risk.py` (B26) | `excess_surprise`, `add_families` (periods) | §9.3 | `test_b26_risk.py` (+) |
| `behavior/incident.py` (B27) | `acc_level_from_p`, `_quiet`, `_pbd` | §9.3 | `test_b27_incident.py` (+) |
| `behavior/governor.py` (B28) | `evidence_factor`, `_ln_arl`, `_observe`, `_level` | §9.3 | `test_b28_governor.py` (+) |
| `behavior/explain.py` (B29), `lib/replay.py` | numeric attribution, neutralisation, cusum / evidence replay, p_replay call | §9.4 | `test_b29_explain.py` (+ a canonical incident: counterfactual valid; fidelity ≤ 1e-6) |
| `behavior/portrait.py` (B30) | `_workload_block`, `_class_bands`, BAND_EXPOSURE_S → per grain, render_zh / render_en | §9.5 | `test_b30_portrait.py` (+) |
| `api/views.py`, `api/routes_v2.py`, `api/routes.py` | `feature_rows`, `_current_workload`, legacy `entity_detail` | §9.5 | `tests/api/test_routes_v2.py` (+ `grains` block, no NaN) |
| `eval/packs.py` | `warmup_phases`; pack_a, b, c, d, smoke | §10 | `tests/eval/test_packs.py` (+): every live ≤ 900 pack's last warm-up phase is at the live cadence and covers ≥ 1 full day of each day type; total span 16 d; smoke has 120 warm-up ticks covering both day types |
| `eval/runner.py`, `eval/metrics.py`, `eval/report.py` | `_baseline_snapshot`, collector, gate 7 / 8 / 11 splits | §9.7 | `tests/eval/test_runner.py`, `test_metrics.py` (+) |
| `pipeline/build.py` | `Runtime.__init__` / `warmup` (plan) | §10 | `test_pipeline_e2e.py` (+ Runtime plan smoke) |

Engines with **no change**: B02 (it reads H descriptors through the
m_baseline default grain; its cold path stays on tick rows), B07, B08, B09,
B10, B11, B12, B13, B23 (the detector list grows through the registry) and
P2 B19–B22 (not registered).

---

## 12. Migration plan: the pipeline stays runnable at every step

Each step is one mergeable change with the full suite green. The mode flag
`ctx.config['grain_mode']` is `'tick'` by default until M8. In tick mode
every grain API degenerates to v2 (D11).

| Step | Content | Must hold |
|---|---|---|
| M0 | `grains.py`, the features grain tables, `SetSketch`, detectors +4 (appended), gating `window_s`, calib keys, store retention, config key. **Before touching engines:** record a golden fingerprint of the mini pack's first 60 ticks in tick mode as `tests/test_tick_mode_golden.py`. It hashes the first 31 columns of behavior.p (the v2 detectors; the appended Q columns stay NaN in tick mode), plus e_day, alarm and incidents. | Suite green. `test_lib_data` updated (31 → 35, deliberate). The golden test passes. |
| M1 | Raw and derived inputs: `*_ids`, `act.slot_events`, `dns_dga_named_n`, think parts (all new series, no existing value changes); D0/D2 slot grid behind `grain_mode == 'canonical'`. | Suite green with no test edits; golden test unchanged; new raw/derived tests. |
| M2 | B01 grain rows (canonical only; tick mode writes nothing new). | `test_b01_grains.py`; property test part A (§13.2). |
| M3 | m_baseline fmt 2 + B03 learners (grain API; tick mode = v2). | B03 suites unchanged in tick mode; `test_b03_grains.py`; golden test. |
| M4 | B04, B05, B06, B14 decision-tick passes; B24 strata; B25 streams, tick types and evidence CUSUMs; B26–B28 periods. | Engine suites (tick mode); the new grain tests; golden test. |
| M5 | m_identity, B15, B16, B17 on H rows; the B24 identity stratum. | B15/B16/B17 suites; `test_b16_grains.py`. |
| M6 | B18, B29, B30, API, eval runner and metrics. | Suites; `tests/api`. |
| M7 | Packs `warmup_phases` + the runtime plan. | `test_packs.py`; mini and smoke e2e. |
| M8 | Flip the default: `grain_mode='canonical'` for pipeline, Runtime, eval and scripts. `tests/helpers.make_ctx` keeps `'tick'` as the default for engine unit tests, whose maths is cadence-agnostic. The e2e tests and the property test run canonical. Update `test_pipeline_e2e.py` expectations that name v2 tick semantics (for example "B01's feature.active is the clock of every learner" becomes feature.meta.<g>). Run `scripts/evaluate.py` for mini, smoke, A, B and E, and update integration.md §8. | Full suite green; eval report. |

**Rollback of the migration:** set `grain_mode='tick'`. That is v2 behaviour
at any step.

---

## 13. New tests

### 13.1 Tolerances (identical traffic, different Δt)

| Class | Tolerance at a common H boundary |
|---|---|
| add | relative 1e-5 (float32 parts) |
| set | exact when every tick has ≤ 256 keys and the union has ≤ 4096; else relative 0.1 (3 HLL σ) |
| map | \|Δ\| ≤ 0.02 in nat when every tick's `__other__` share is ≤ 5 %; otherwise skipped (top-64 truncation per tick; §16) |
| span: periodicity, timing_regularity, duty_cycle | exact (slot grid from the same events) |
| span: req_per_session | relative 0.1 (stream cap at 3600 s) |
| think_time | relative 0.1 when `act.stream_frac` = 1 in every tick |

### 13.2 Cadence-invariance property test (`tests/test_cadence_invariance.py`, new)

**Part A (fast, in the suite, ~10 s).**
1. Generate 2 human and 1 machine persona for one local day at Δt = 60 in
   non-aggregated mode (`TrafficGenerator.step(60, aggregated=False)`),
   keeping every Observation with its real `ts`.
2. **Re-batch** the same observations into ticks of 900 and 3600 s. A tick
   ending at t gets every observation with ts ∈ (t − Δt, t].
3. Run raw + derived + B01 (canonical) on the three tick sequences.
4. Assert, at every common H boundary: `feature.nat.h` equal within §13.1,
   `feature.meta.h` cov = 3600 and active equal, and `feature.sketch.h`
   equal within 1e-6 when the maps are untruncated.
5. At Δt = 60 vs 900: `feature.nat.q` equal at every common Q boundary.
6. **Property form** (hypothesis-style, seeded, 20 cases): random event
   times and sizes; for each Δt ∈ {60, 300, 900, 3600} and misaligned tick
   offsets, the add-class H values equal the direct computation over the
   event list.

**Part B (slow marker, nightly, ~3 min).**
1. 6 control personas (4 human, 2 machine, no scenarios). Warm-up by the
   §10 rule for a Monday start: 96 × 3600 + 288 × 900 (Fri–Sun).
2. Then 12 live hours (09:00–21:00) at Δt = 60. The twin run re-batches the
   same live observations at 900 s.
3. Assert, in canonical mode:
   - (i) H rows of the two live runs are equal within §13.1;
   - (ii) single-tick alarm ticks (e_day ≤ 0.03), summed over entities, are
     ≤ 2 in each run. The expectation is 0.09 over 3 entity-days;
     P(≥ 3) < 2e-4 under Poisson;
   - (iii) no incident ≥ MEDIUM, and no `unknown_identity` or
     `identity_mismatch` ≥ MEDIUM;
   - (iv) no B14 `acc_alarm`;
   - (v) FAR ≥ LOW(60) / FAR ≥ LOW(900) ∈ [0.5, 2] **or** both are 0;
   - (vi) the per-detector p of the H stream at common H ticks agrees
     between the two runs within |Δ log10 p| ≤ 0.5 (same rows; the
     remaining differences come from T-stream trust and from B24 ring
     admission timing).
4. A third run keeps v2 tick mode on the same traffic and is expected to
   **fail** (ii)–(iv). This is recorded as an xfail sentinel, so the test
   demonstrably measures the defect.

### 13.3 Other new tests

The new tests per engine are listed in §11 (`test_grains`,
`test_features_grains`, `test_b01/b03/b04/b14/b15/b16/b24/b25_grains`,
`test_packs`, `test_tick_mode_golden`). The existing timing and perf tests
(B16 edges, calib perf) are unchanged in tick mode.

---

## 14. Expected effect on each failing gate

The baselines are integration.md §8 (A and B after the §4 fixes) and the
900-s-warm-up experiment, which removes the cadence switch but also the 14 d
of history. "Removes" names the mechanism that goes away. The numbers are
expectations to be verified in M8, not promises. Residual causes are named
with their §8 item.

| Gate | Now (A / B) | Mechanism removed by v2.1 | Expected after M8 | What remains |
|---|---|---|---|---|
| 1 Detection | 1/12, 1/10; loud 0.125 | Controls latched from the first live hours (distinct-count +2 zr, K-tick identity, empty cc rings), so incidents never close and threats only escalate old incidents | ≥ experiment level (7/12 detected, 5/12 in deadline). With 14 d of H history, the midpoint fix, and Q providing 15-min latency for loud scenarios, the loud recall should improve over the experiment | B06 SPE on machine personas (§7.2); B24 pm prior for sparse entities; warm-up extremes in B25 meta rings (§8 list) |
| 2 TTD | loud TTD misses | The Q stream scores every 900-s tick; E (60 s) has Q decisions every 15 min on the same wall-clock boundaries as A's ticks, so E ≤ A + 15 min by construction | loud ≤ 2 ticks where detected | detection itself (gate 1) |
| 3 FAR | ≥ LOW 0.126, ≥ MED 0.096, HIGH+ 23, CRIT 11 | Identity p = 1e-38 on the first live ticks; latched cusum / mcusum; cc-reset rings | HIGH+ and CRITICAL fall sharply (their triggers were the cadence artefacts). FAR ≥ LOW / MEDIUM may **rise** at first, because incidents now close and reopen, as in the experiment (0.37) | §8 items; B25 meta-ring warm-up admission |
| 3 cadence-invariance check | not computable | e_day budget exact per tick type; identical H rows | defined, expected within [0.5, 2] | tick-native detectors (§9.6) |
| 4 Legit | 0.42 | L-type changes landed on latched entities | modest improvement | governor acceptance paths |
| 5 Burden | 5 notifications per TP; 2 incidents per episode | FP incidents opened before threats and reopened | incidents per episode → 1 | notification policy |
| 6 Poisoning: null KS after the attack | 0.27 | Rings keyed by cc | lower | B24 ring gating |
| 7 Calibration | e_day ≤ 0.03 exceedance 262×; change path 0.317 per day; temporal_categorical 0.598; evidence 0.143 | The zr offset latching k = 0.25 charts; H rings valid across the switch; charts h now per hour; the pack's live-cadence warm-up fills the cc-900 T rings | Exceedance at most the experiment (18×) for grain detectors; the change path near its 0.02–0.04 budget; temporal_categorical toward the experiment's 0.118 (pack fix); evidence ≤ 0.045 by the stream split | KS D 0.82 (point masses of accumulators, e.g. creep D = 1.0 at p_eq = 1: §8); jsd / novelty_rate tick-size dependence |
| 8 Identification | top-1 0.98; EER / T9b pending | Windows like-for-like; the unknown_identity flood on enrolled controls disappears (runtime smoke: 16 incidents after 16 ticks → ~0 expected) | T9b power can finally be judged (§8 note) | T99 calibration of humans |
| 9 Classes | ARI 0.50 (experiment 0.81) | Descriptors from H anchors covering both day types (the smoke fix and the warm-up rule) | toward the experiment level | HDBSCAN stability (tuning) |
| 10 Explanation | hit@3 0.5; CF validity 0.5 | Spurious distinct_* z no longer top the attribution; per-grain replay | improvement | B29 replay fidelity |
| 11 Portrait coverage | 0.965 (too wide) | Bands at each grain's own exposure and dispersion (v2 took 15-min bands from mixed-exposure fits) | toward 0.90 | — |
| 14 CPU | 1460 s per pack-seed; live p95 843 ms | B04 at 60 s scores 1/12 as often; B16 cheap modalities once per H row; B03 commits hourly | Packs: +20–35 % warm-up ticks (§10), B04 +25 % mean per tick at 900 s (H every 4th tick), p95 up on H ticks. **Not fixed**; open issue 5 (vectorised NB / BB kernels) remains the lever | — |
| 15 Robustness | pass | The stale checker already uses the per-series period | pass | — |

---

## 15. Costs

- **CPU per tick at 20 entities** (measured v2 numbers from integration.md
  §6, extrapolated):
  - **B01:** +0.05 ms per entity per tick for parts and live sums. On
    decision ticks, set unions, map merges and sketch.h add ~0.3–0.5 ms per
    entity, which is +6–10 ms on H ticks.
  - **B04:** at 900 s, Q each tick (≈ v2) plus H every 4th tick (≈ 2× on H
    ticks); at 60 s, ≈ 1/12 of v2 per tick on average; at 3600 s, ≈ v2.
  - **B06:** similar to B04.
  - **B14:** 1/4 of v2 at 900 s.
  - **B16:** per H row instead of per tick.
  - **B03:** fewer commits (hourly H rows + Q rows).
  - **Global boundaries** concentrate the H work on one tick per hour. The
    p95 at 900 s rises by roughly the B04 + B06 per-grain cost (+~40 ms at
    20 entities).
- **Memory per entity at 60 s:**
  - feature.part: 120 × 47 × 4 B ≈ 23 KB;
  - feature.live.h and .q: 2 × 360 × 208 B ≈ 150 KB;
  - nat and meta rings for H and Q over 8 d: ≈ 200 KB;
  - raw `*_ids`: ≤ 75 × 4 × 2 KB ≈ 0.6 MB worst case;
  - behaviour H rings (4 d, hourly): < 100 KB.

  The total is < 1.1 MB, inside the 12 MB per-entity gate.
- **Packs:** +96 to +288 ticks per pack-seed at ~0.26 s per tick is +25 to
  +75 s.

---

## 16. Risks, open items, rejected alternatives

**Risks and open items**

1. **Transfer error before native Q support.** Mitigations:
   - the default v = m is an upper bound for non-negative within-hour
     correlation;
   - provisional scores get their own B24 strata and half weight, and
     cannot count toward HIGH;
   - B24 KS health halves a detector whose realised rate exceeds 2× budget.

   Residual: negatively correlated quarters (alternating bursts) can give
   v > m. ω is clipped at 2m, so v ≤ 7.
2. **Top-64 truncation** makes the four map features only approximately
   cadence-invariant when a tick has more than 64 named values. Remedy if
   needed: per-key id sketches for maps, or TOP_K 256 for the three map
   sources.
3. **HLL error** (3.25 %) above 4096 distinct per window, for scanners and
   high-fanout hosts. This is acceptable for their NB scale.
4. **Tick-native detectors** keep cc strata (§9.6). Follow-up: jsd and
   novelty_rate on H-grain token distributions, and seq/dwell normalised per
   token.
5. **H evidence latency** is ≤ 1 h at Δt < 3600. Q covers bursts at 15-min
   latency. Loud H-only evidence (for example a distinct-count surge that
   is Q-native-only for a new entity) can lag by up to one hour.
6. **Q-native-only features for new entities in live** are unscored at Q
   for about one day. Their H scoring is immediate through class backoff.
7. **DST:** row_tctx at the midpoint and local-aligned spans handle the
   23- and 25-h days. The pack C scenario covers it.
8. **Tick mode must stay bit-identical** during migration. It is guarded
   by the golden test (M0).

**Rejected alternatives**

- **Baselines keyed by cadence class** (integration §8 suggestion).
  Rejected because:
  - every cadence needs its own warm-up;
  - it does not fix identity windows or distinct-count semantics;
  - it multiplies models by 4.
- **An AR(1) or effective-sample correction on rolling rows.** The overlap
  structure is MA(m − 1), which AR(1) mis-whitens. Thresholds would depend
  on Δt through m. Rejected in favour of decimation (D7).
- **A cadence-aware normaliser for distinct counts** (for example a
  Heaps-law exponent). It is entity-specific, drifts with behaviour, and is
  unnecessary once unions are exact.
- **Per-entity grain phases.** They break the synchronous cross-entity rows
  of B05 and B18 (D3).
- **Q as the primary grain.** Q is not observable at 3600 s (warm-ups,
  pack D).
- **Scoring rolling rows every tick with Bonferroni e_day** (`p·86400/Δt`).
  It is valid, but it costs H evidence 60× at Δt = 60 and still needs
  decimation for the charts.

---

## 17. Implementation notes (implementer)

Where the code differs from the design above, and why. Each item is a
correction of the design, not a tuning.

**Representation (§3–§4)**
- `think_log_sum` / `think_gaps` are emitted by D2 in **canonical mode
  only** (the design said both modes). In tick mode they changed D2's
  `produces` list and so v2's output (golden test); `test_d2_session`
  now lists them (deliberate).
- `compute_parts` clips negative sums to 0 except `p_think_ls` (a sum of
  log gaps is legitimately negative).
- R2's canonical new-template rule: a token counts as new for
  `new_template_ratio` while its event is within `NEW_WINDOW_S` = 900 s of
  the token's first sighting (`model['tok_first']`), so the ratio is a
  wall-clock quantity (v2's "first seen in this tick" depends on Δt).
- `distinct_templates` is exact on the templater's output; the templater
  (Drain-lite) itself masks a few tokens differently depending on batch
  interleaving, so the property test allows max(1, 15 %) for that feature.
- B01 coverage of a grain window is taken from B01's own tick log (so the
  first tick of an entity is covered) with the ring as a fallback; D2's
  slot-grid duty starts its coverage at the first event's slot.

**Baselines (§6)**
- The paired-hour statistic b uses the H **predictive mean** (not
  λ̂²/r_H of the fitted row): the fitted value is not available when the
  pair is folded, and the predictive mean is what the transfer is applied
  to. `paired_stats` broadcasts b to a's shape.
- The Q pseudo-rows reproduce the transferred predictive exactly in its
  moment overdispersion; the NB size the pseudo-rows give is
  1/(1/r_T + 1/(κ_T·15μ)) (the Gamma shape of κ_T rows), which is what a
  conjugate posterior on κ_T rows must give (`test_b03_grains`).
- B18 learns its anchors from `behavior.class.agg.h` (written on H ticks)
  while `behavior.class.agg` stays per tick; its aggregate bands are per
  hour in canonical mode (`exposure_s` 3600).
- B30 bands: per grain under `workload.<f>.grains.{h,q}`; the H band is the
  H anchor's own predictive at 3600 s, the Q band its transfer (σ15 × v15),
  `provisional` from the native Q anchor's share (< 0.5).

**Detectors and calibration (§7–§9)**
- `DETECTOR_INFO[d]['strata']` keeps the v2 labels; the grain strata are
  a new key `grain_strata`, so v2 code reading `strata` is unchanged.
  `test_lib_data::test_detector_families_partition` changed with the
  registry (the four Q detectors are in the intensity / shape families;
  deliberate, like the N_DETECTORS 31 → 35 change).
- Tick mode keeps v2's scheduling keys in B04 / B05 / B06 (the grain passes
  add the grain to the key only in canonical mode); otherwise the B06 edge
  test's refit timing moved.
- B24 pending entries are tuples in canonical mode; `_ensure_layout`
  preserves them (an `int()` of the tuple crashed the JSON round trip).
- B15 / B17 run on H ticks only in canonical mode (B17's identity windows
  are the same 4 active H rows as B16's).
- B25 audits each stream: S_t against 1/66 d on its ticks, S_h on
  `behavior.q_inst.h` (one row per hour) with its own `h_mult_h`.
- B29 recomputes, in canonical mode, the grain rows scored at the replayed
  tick (H row with exposure = coverage and midpoint tctx; Q row with
  `predictive_q`, `pf.q`, `zi.q`, `model.density.q`), p through
  `p_replay(grain, prov)` with B24's recorded grain dayparts, the meta
  strata B25 recorded in its pending entry (else rebuilt), e_day = q_all ×
  n_τ/β_τ, the T-stream and H-stream evidence CUSUMs over their own
  excursions, and the CUSUM bank over ≤ 96 H states. Attributions use the
  grain whose detector holds the smallest p (Q when a `_q` detector
  drives), with values per grain exposure (`/h`, `/15 min`).
- The eval stale checker treats a grain series of a re-activated entity as
  due only after two periods of activity (`_StaleChecker._due_since`).
- **B14 holds its latches between H ticks** (design gap: §7.2 says the chart
  does not move between H ticks, but not what B14 reports there). A latched
  cusum / mcusum / bocpd / creep alarm is re-emitted on every Q / T tick
  until the next H tick decides; without it B25's accumulator path, B27 and
  the eval saw an on-off train at the H period (pack A: 71 "change"-path
  onsets on control entities, incidents closing and reopening).
- **B25 persists the S_h reset at the first live tick** (found in pack A and smoke:
  S_h is written on H ticks only, so a reset at a non-H first live tick was
  lost and the warm-up S_h, several hundred, resumed on the next H tick:
  in smoke, 7 of 9 machine personas' H-stream evidence alarmed from the first live hour).
- Eval: the runner records the full contract-I config (so a run states its
  grain mode); metrics gate 7 splits exceedance by tick type (expected
  count x·β_τ/n_τ per tick), gate 11 adds H-row and Q-row coverage against
  the portrait's per-grain bands. Gate 8's "per H row top-1" is not added:
  B15 publishes no per-row recall (the per-tick figure is n/a already).

**Packs and runtime (§10)**: as designed. `Runtime(warmup_ticks=n)` keeps
the v2 plan n × 900 s (API tests, `APPMON_WARMUP_TICKS`); the default plan is
[(120, 3600), (192, 900)] and a warning names a last warm-up phase that
misses a day type.

**Tests (§13)**: existing expectations changed only where the spec changed
them (deliberate): `test_lib_data` (35 detectors, Q detectors in the
intensity / shape families), `test_d2_session` (D2 `produces`), and
`tests/eval/test_packs.py::test_pack_timelines` (the §10 warm-ups: A 984,
B 1824, C 1416, D 1104 ticks). `test_pipeline_e2e` needed no change after
the flip. Engine / lib / core unit tests run tick mode through
`tests/{engines,lib,core}/conftest.py` and `helpers.ctx` / `run_engine`
(also at collection time, for module-level `default_config()` calls). New:
`test_grains`, `test_features_grains`, `test_b01_grains`, `test_b03_grains`,
`test_grain_inputs`, `test_tick_mode_golden`, `test_cadence_invariance`,
`test_grains_pipeline` (a canonical mini run checking B04 / B06 / B14 /
B24 / B25 / B29 / B30 / runner per grain), the Runtime plan and pack-rule
tests, and the API `grains` block. Part A as designed. Part B's live phase is generated
non-aggregated, so its warm-up is too (aggregated generator records change
the stream-timestamp and session features). Part B asserts the twin-run
equalities; the absolute null levels (ii)–(iv) are an `xfail` with the
reason below.

**Open (found while verifying)**
1. **B11 timing p on short warm-ups.** B24 scores `timing` from its
   (daypart, cc) ring, and below 64 entries blends in B11's own pm, a G test
   (2N·JSD / φ) whose overdispersion is under-estimated: pm reaches the
   float32 floor (1e-38) on ordinary Monday traffic of a human persona
   (JSD 0.22–0.29 live vs 0.13–0.19 in warm-up). In Part B the cc-900
   `timing@wd_day` ring holds 42 entries after a Fri–Sun 900-s warm-up
   (humans idle at the weekend), so p ≈ 2e-32 on every live tick of every
   human persona, in tick mode too. This dominates Part B's single-tick
   count. It is the B24
   small-sample prior issue of integration.md §8, not a grain issue. Fix
   candidates: a conservative pm for B11 (φ from held-out rows), or no pm
   prior for tick-native detectors whose pm is not calibrated.
2. **B06 on a warm-up shorter than D_min + 14 d.** The density commits zi
   rows computed while B04's baseline still predicted from the hyperprior
   (the first D_min_s of a young entity: zi ≈ −2.5 on every rate feature),
   so after the switch to own predictives T² ≈ 2000 on every row (mini pack,
   tick mode identical). In the 16-d packs those rows are a small share of
   the ≤ 336 committed rows. Fix candidate:
   B06 admits a zi row only when B04 scored it against an own-support
   predictive.

### 17.1 Status after evaluation rounds 2–3 (docs sync, 2026-09-28)

Changes in canonical mode since §17 (each with a regression test that fails
with the fix switched off; tick mode and its golden are unchanged except where
noted in integration.md §8.2):
- **B14:** the end-of-warm-up chart restart runs on the first live tick
  whatever its type (it ran on the first live H tick only, so a warm-up latch
  was reported as a live alarm for up to an hour: 59 ticks at 60 s).
- **B18 class_rhythm:** per-slot membership comes from `act.slot_events`
  (a member active anywhere in a 3600-s tick used to count as active in every
  slot of it), every complete slot is scored and learned in time order, and a
  slot is scored only when its own bin holds ≥ 3 decayed slots
  (`RHYTHM_MIN_BIN_W`) instead of falling back to the pooled prior of the
  other bins.
- **lib-4 `signature.rule_match`:** additive counter clauses are read as totals
  per 15-min grain (`ADDITIVE_COUNTERS`, trailing 900 s pro rata at Δt < 900,
  × 900/Δt at Δt ≥ 900), so thresholds keep their 900-s meaning. Ratio,
  entropy and concentration clauses are still read per tick (open item 3
  below).
- Round 2 (commit fdcde68): continuous-time human sessions in the generator,
  B13's counting-noise floor, B24's own-history floor, B27 quiet-close and
  lib-4 evidence rules, B11 undefined descriptors.

State of the two open items above:
1. **B11 / tick-native pm on short strata** is reduced, not solved: B24's pm
   prior randomises the two atoms of pm (`m_calib.pm_prior`, integration.md
   §8.2) and the own-history floor stops a prior from contradicting the ring
   (§10.2 #5), but a live score beyond every ring entry still takes pm. After
   the Runtime's or pack E's 900 → 60 s switch the (daypart, t, cc = 60) meta
   strata and the cc = 60 B24 strata start empty, p_all includes the
   accumulators, and the evidence cap keeps high-accumulator live rows out of
   the rings: 6.7 % of control t-ticks raise a single-tick alarm (budget_vol,
   timing, jsd, budget_breadth). This is why the APPMON_SLOW Part B test
   `test_part_b_live_60_equals_900` now FAILS (H rows and B14 alarm episodes
   are equal, but 187 single-tick alarms at 60 s against 3 at 900 s). It
   passed during M8 only because the old generator made both twins equally
   noisy. It is left failing, not xfail. Candidate fixes: seed cc = 60 strata
   from cc = 900 for the window statistics, or keep accumulators out of the
   t-tick single-tick fusion (integration.md §10.7 item 2).
2. **B06 on short warm-ups:** unchanged (engines.md B06).

Further open items that belong to this design:
3. lib-4 per-tick ratio / entropy / concentration clauses at 60 s (new
   categories on control entities feed B08's category dimension); evaluating
   lib-4 on Q-window aggregates needs derived-engine support.
4. Loud TTD ≤ 2 ticks at 900 s is out of reach while Q-grain p stays at
   1e-3 … 3e-4 under the default transfer v = 4 (no paired hours before the
   first live day); the β shares are not the lever (integration.md §10.5).
5. Memory at 60 s: age-based retention of per-tick dict series (behavior.score
   / pm / p, 1 d) holds 15× the points of 900 s; pack E extrapolates to about
   163 MB per entity over 8 days (gate 14: 12 MB).

Measured (final round-3 tree, integration.md §10.4): FAR(E @ 60 s) /
FAR(A @ 900 s) = 3.10 (gate 3 band [0.5, 2]); single-tick exceedance at
e_day ≤ 0.03 is 64× nominal at cc 900 and 2 818× at cc 60 (gate 7 band
[0.5, 2]).
