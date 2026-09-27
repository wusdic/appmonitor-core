BACKEND (backend/app/api/routes.py)

The routes are read projections, plus a small set of write endpoints for the analyst loop. All times are shown in ctx.config.tz.

Migrating the legacy endpoints (these must not regress):
- GET /api/systems/{s}/entities (routes.py:99-124)
  - Rank by behavior.risk, not by the latest event score.
  - Replace the 45-second cutoff on time.time() with store time.
  - Return: risk, tier, trend, class_path, role name and prob, separability, open incident count, regime state.
  - Keep these legacy fields, now computed differently:
    - anomaly_score = min(1, log10(1/max(e_day, 1e-12))/8), from the latest q_all;
    - drift_score = max over {cusum, mcusum, bocpd, creep} of min(1, score/10);
    - archetype = class_path;
    - separability = the redefined value.
  - This keeps frontend/js/app.js:108-116 working. It no longer searches for events with kind 'anomaly' or 'drift' (routes.py:109-110).
- GET /api/systems/{s}/entities/{e}
  - Returns the profile, portrait summary, risk, identity, continuity and regime.
  - features[] now carries current, predictive p5/p50/p95 in natural units, and the engine's own z and zr. This replaces the unfloored z recomputed at routes.py:140.
  - archetype, archetype_confidence and separability keep their semantics as documented above (routes.py:147-148).

New endpoints:
- Entity:
  - GET /api/systems/{s}/entities/{e}/portrait?version=
  - GET /api/systems/{s}/entities/{e}/portrait/diff?from=&to=
  - GET /api/systems/{s}/entities/{e}/timeline?since= (indexed store.timeline)
  - GET /api/systems/{s}/entities/{e}/scores?since=: p per detector, p_family, q_inst, q_all, e_day, evidence, trust, trust_prov and regime series.
  - GET /api/systems/{s}/entities/{e}/identity: confusion row, traits, recall and EER with CI, T99, attribution posterior series.
- Class:
  - GET /api/systems/{s}/classes: role, sub, static and pool classes, each with members, risk and open class incidents.
  - GET /api/systems/{s}/classes/{id}: class portrait, class_monitor bands, active-fraction heatmap, adoption history, identifiability, lineage, version.
- Incidents:
  - GET /api/incidents?system=&entity=&status=&severity=&kind=entity|class
  - GET /api/incidents/{id}: evidence per family and axis, e_day, explanation, counterfactual (validity and scope), narrative, campaign, and parent/child links for common mode.
- Feedback (writes Labels):
  - POST /api/incidents/{id}/feedback and POST /api/events/{id}/feedback, with body {verdict, scope, ttl_s, note}.
  - GET /api/label-queue
- Operations:
  - GET /api/detectors/health returns:
    - per engine: last_run, last_error, error_count, last_error_ts, and the staleness of each series it produces (now − last_write_ts);
    - per detector and system: KS D, realised rate against the e_day budget, weight_mult, degraded share.
  - GET /api/eval/report
  - GET /api/store/memory (memory_report)

FRONTEND (frontend/, vanilla JS plus inline SVG)

1. Entity page:
   - portrait card with narrative;
   - 7×24 rhythm heatmap (P_active per 15-minute slot, aggregated by hour) with current activity overlaid;
   - workload bands: p5/p50/p95 in natural units per day type, with the current value;
   - client-stack chips; top templates and destinations; n-gram chips; distinctive traits;
   - identifiability gauge with CI and the confusable list;
   - risk timeline with incident markers, and regime, version and rollback markers.
2. Class page:
   - role and sub hierarchy tree; members with membership bars;
   - class portrait; class aggregate bands; active-fraction heatmap;
   - adoption history with risk flags; class-vs-system distinctive tokens; outlier members; class incidents.
3. Incident queue:
   - severity with the e_day explained ('as rare as once in N days for this entity');
   - entities or class; narrative; per-family and per-axis bars; new tokens;
   - counterfactual with its scope;
   - feedback buttons (tp, fp, expected change, benign known) with scope.
4. System view:
   - common-mode and coherent shifts; campaign graph;
   - detector health panel with engine errors and staleness.
5. Eval report page.

Update frontend/js/app.js:108-116 and 134-135 to show tier and risk in place of the raw anomaly percentage. The legacy fields remain available for one release.