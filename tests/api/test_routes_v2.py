"""API v2 (docs/lib3/api_ui.md) against a briefly warmed real Runtime.

One module-scoped Runtime (short 900-s warm-up + a few 60-s live ticks, as in
scripts/smoke.py) serves every test: each endpoint must answer 200 with its
documented top-level keys, unknown keys must 404, the legacy entity fields
must survive, and a feedback POST must create a Label that B23 feedback
consumes on the next tick. Also checks the frontend JS parses (node --check).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from asgi_client import make_client  # noqa: E402

from app.api import routes  # noqa: E402
from app.main import app  # noqa: E402
from app.models.schema import Incident, Severity  # noqa: E402
from app.pipeline.build import Runtime  # noqa: E402

WARMUP = int(os.environ.get("APPMON_TEST_WARMUP", "76"))
LIVE = 3
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


@pytest.fixture(scope="module")
def rt():
    r = Runtime(warmup_ticks=WARMUP, live_period_s=0.0, seed=7)
    r.warmup()
    for _ in range(LIVE):
        r.step_once()
    prev = routes.RUNTIME
    routes.RUNTIME = r
    yield r
    routes.RUNTIME = prev


@pytest.fixture(scope="module")
def client(rt):
    return make_client(app)


@pytest.fixture(scope="module")
def target(rt):
    """(system, entity) of the entity with the richest profile."""
    st = rt.store
    best = None
    for s in st.systems():
        for e in st.entities(s):
            p = st.profile(s, e)
            if p is not None and (best is None or p.sample_count > best[2]):
                best = (s, e, p.sample_count)
    assert best is not None, "warm-up produced no profile"
    return best[0], best[1]


def _get(client, path, status=200):
    resp = client.get(path)
    assert resp.status_code == status, (path, resp.status_code, resp.text[:400])
    return resp.json()


def _has(d, keys):
    missing = [k for k in keys if k not in d]
    assert not missing, f"missing keys {missing} in {sorted(d)}"


def _no_nan(obj):
    """Payloads must be strict JSON (Starlette forbids NaN): walk and check."""
    if isinstance(obj, float):
        assert math.isfinite(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _no_nan(v)
    elif isinstance(obj, list):
        for v in obj:
            _no_nan(v)


# ------------------------------------------------------------------ legacy
def test_health_carries_tz_and_store_time(client, rt):
    d = _get(client, "/api/health")
    _has(d, ["status", "warmed", "live_ticks", "tick_count", "tz", "now", "now_local"])
    assert d["tz"] == rt.config["tz"]
    assert abs(d["now"] - rt.gen.vt) < 1e-6          # store time, not wall clock
    assert d["now_local"].endswith("+08:00")          # Asia/Shanghai


def test_entities_ranked_by_risk_with_legacy_fields(client, target):
    s, _ = target
    d = _get(client, f"/api/systems/{s}/entities")
    _has(d, ["system", "tz", "now", "entities"])
    assert d["entities"]
    for row in d["entities"]:
        _has(row, ["entity", "archetype", "archetype_confidence", "separability", "stable",
                   "sample_count", "anomaly_score", "drift_score", "current_activity",
                   "current_category",                                     # legacy
                   "risk", "tier", "trend", "class_path", "role", "role_name", "role_prob",
                   "open_incidents", "regime"])                             # v2
        assert 0.0 <= row["anomaly_score"] <= 1.0 and 0.0 <= row["drift_score"] <= 1.0
        assert row["tier"] in (None, "low", "medium", "high", "critical")
    risks = [x["risk"] for x in d["entities"] if x["risk"] is not None]
    assert risks == sorted(risks, reverse=True)


def test_entity_detail_features_and_legacy(client, target):
    s, e = target
    d = _get(client, f"/api/systems/{s}/entities/{e}")
    _has(d, ["system", "entity", "archetype", "archetype_confidence", "separability", "stable",
             "sample_count", "updated", "features", "seasonal", "events", "matches",   # legacy
             "risk", "portrait", "identity", "continuity", "regime", "peer_group", "tz"])
    assert d["features"]
    f0 = d["features"][0]
    _has(f0, ["name", "current", "p5", "p50", "p95", "z", "zr",
              "baseline", "spread", "stable"])                               # legacy names kept
    _no_nan(d)


def test_unknown_entity_and_system_404(client, target):
    s, _ = target
    _get(client, f"/api/systems/{s}/entities/203.0.113.250", status=404)
    for sub in ("portrait", "portrait/diff", "timeline", "scores", "identity"):
        _get(client, f"/api/systems/{s}/entities/203.0.113.250/{sub}", status=404)
        _get(client, f"/api/systems/no-such-system/entities/x/{sub}", status=404)
    _get(client, "/api/systems/no-such-system/classes", status=404)
    _get(client, f"/api/systems/{s}/classes/no-such-class", status=404)
    _get(client, "/api/incidents/inc-does-not-exist", status=404)
    _get(client, "/api/systems/no-such-system/summary", status=404)


# ------------------------------------------------------------------ entity v2
def test_entity_portrait(client, target):
    s, e = target
    d = _get(client, f"/api/systems/{s}/entities/{e}/portrait")
    _has(d, ["system", "entity", "version", "current_version", "versions", "portrait",
             "rhythm", "current", "risk", "tz"])
    _has(d["portrait"] or {"json": 0, "text_zh": 0, "text_en": 0, "diff": 0},
         ["json", "text_zh", "text_en", "diff"])
    if d["rhythm"] is not None:
        assert len(d["rhythm"]["p_active"]) == 7 and len(d["rhythm"]["p_active"][0]) == 24
    if d["versions"]:
        v = d["versions"][-1]["version"]
        d2 = _get(client, f"/api/systems/{s}/entities/{e}/portrait?version={v}")
        assert d2["version"] == v
    _get(client, f"/api/systems/{s}/entities/{e}/portrait?version=987654", status=404)
    _no_nan(d)


def test_entity_portrait_diff(client, target, rt):
    s, e = target
    resp = client.get(f"/api/systems/{s}/entities/{e}/portrait/diff")
    if not rt.store.profile_versions(s, e) and not (
            rt.store.profile(s, e).extra or {}).get("portrait"):
        assert resp.status_code == 404
        return
    assert resp.status_code == 200, resp.text[:300]
    _has(resp.json(), ["system", "entity", "from", "to", "diff", "text_from", "text_to"])


def test_entity_timeline_scores_identity(client, target):
    s, e = target
    d = _get(client, f"/api/systems/{s}/entities/{e}/timeline")
    _has(d, ["system", "entity", "since", "items", "regime_markers"])
    for it in d["items"]:
        _has(it, ["ts", "ts_local", "type"])
    d = _get(client, f"/api/systems/{s}/entities/{e}/timeline?since=0")
    assert isinstance(d["items"], list)

    d = _get(client, f"/api/systems/{s}/entities/{e}/scores")
    _has(d, ["detectors", "p", "p_family", "q_inst", "q_all", "e_day", "evidence",
             "trust", "trust_prov", "regime", "risk"])
    assert len(d["q_all"]["ts"]) == len(d["q_all"]["values"])
    for det, vals in d["p"]["values"].items():
        assert det in d["detectors"] and len(vals) == len(d["p"]["ts"])
    _no_nan(d)

    d = _get(client, f"/api/systems/{s}/entities/{e}/identity")
    _has(d, ["fitted", "recall1", "recall1_ci", "eer_hard", "eer_ci", "t99", "confusion_row",
             "confusable_with", "traits", "posterior"])
    _has(d["posterior"], ["ts", "posterior_self", "p_unknown"])


# ------------------------------------------------------------------ classes
def test_classes(client, target, rt):
    s, _ = target
    d = _get(client, f"/api/systems/{s}/classes")
    _has(d, ["system", "classes", "tree", "system_risk"])
    for c in d["classes"]:
        _has(c, ["key", "kind", "members", "n_members", "risk", "open_incidents"])
        assert c["kind"] in ("role", "sub", "static", "pool")
    keys = [c["key"] for c in d["classes"]]
    if not keys:                                       # warm-up too short to type roles
        pytest.skip("no classes after the short warm-up")
    for key in keys[:3]:
        dd = _get(client, f"/api/systems/{s}/classes/{key}")
        _has(dd, ["key", "kind", "members", "portrait", "bands", "active_frac_heatmap",
                  "adoption", "identifiability", "lineage", "version", "incidents"])
        _no_nan(dd)


# ------------------------------------------------------------------ incidents
@pytest.fixture(scope="module")
def incident_id(rt, target):
    """A live incident of the run, or a synthetic one put through the store
    API (a short warm-up may not have opened any)."""
    st = rt.store
    live = st.incidents(status=("open", "acked"))
    if live:
        return live[0].id
    s, e = target
    now = rt.gen.vt
    return st.put_incident(Incident(system=s, entity=e, entities=[e], kinds=["alarm"],
                                    axes=["volume"], opened=now - 600, last_seen=now,
                                    severity=Severity.MEDIUM, e_day_min=1e-3, risk=40.0,
                                    evidence=[{"ts": now, "source": "alarm", "families":
                                               ["intensity"], "axes": ["volume"],
                                               "e_day": 1e-3}]))


def test_incident_queue_and_detail(client, incident_id):
    d = _get(client, "/api/incidents")
    _has(d, ["count", "counts", "incidents"])
    assert any(i["id"] == incident_id for i in d["incidents"])
    for flt in ("status=open", "severity=low,medium,high,critical", "kind=entity", "kind=class"):
        _has(_get(client, f"/api/incidents?{flt}"), ["incidents"])
    _get(client, "/api/incidents?kind=bogus", status=422)
    d = _get(client, f"/api/incidents/{incident_id}")
    _has(d, ["id", "system", "entity", "severity", "e_day_min", "rare_once_in_days",
             "evidence", "evidence_by_family", "evidence_by_axis", "explanation",
             "counterfactual", "narrative", "campaign", "parent", "children", "labels"])
    _has(d["counterfactual"], ["set", "valid", "scope"])
    _no_nan(d)


def test_feedback_round_trip_creates_label(client, rt, incident_id):
    st = rt.store
    n0 = len(st.labels())
    resp = client.post(f"/api/incidents/{incident_id}/feedback",
                       json={"verdict": "fp", "scope": "pattern", "ttl_s": 3600,
                             "note": "api test"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    _has(body, ["ok", "label_id", "label"])
    labels = st.labels()
    assert len(labels) == n0 + 1
    lb = labels[0]
    inc = st.get_incident(incident_id)
    assert (lb.id, lb.target_type, lb.target_id, lb.verdict, lb.scope) == \
        (body["label_id"], "incident", incident_id, "fp", "pattern")
    assert (lb.system, lb.entity) == (inc.system, inc.entity)
    assert lb.ts == pytest.approx(rt.gen.vt)                   # store time
    # visible in the label queue and in the incident
    q = _get(client, "/api/label-queue")
    _has(q, ["queue", "governor", "labels"])
    assert any(x["id"] == lb.id for x in q["labels"])
    assert any(x["id"] == lb.id for x in _get(client, f"/api/incidents/{incident_id}")["labels"])
    # B23 feedback consumes it on the next tick (profile.extra.feedback)
    rt.step_once()
    fb = (st.profile(inc.system, inc.entity).extra or {}).get("feedback") or {}
    assert fb.get("last_verdict") == "fp" and fb.get("n_labels", 0) >= 1


def test_event_feedback_and_validation(client, rt, incident_id):
    st = rt.store
    evs = st.events(limit=1)
    if evs:
        resp = client.post(f"/api/events/{evs[0].id}/feedback",
                           json={"verdict": "benign_known", "scope": "entity"})
        assert resp.status_code == 200, resp.text
        assert st.labels()[0].target_type == "event"
    assert client.post("/api/events/ev-nope/feedback",
                       json={"verdict": "tp"}).status_code == 404
    assert client.post(f"/api/incidents/{incident_id}/feedback",
                       json={"verdict": "maybe"}).status_code == 422
    assert client.post(f"/api/incidents/{incident_id}/feedback",
                       json={"verdict": "tp", "scope": "galaxy"}).status_code == 422
    assert client.post("/api/incidents/inc-nope/feedback",
                       json={"verdict": "tp"}).status_code == 404


# ------------------------------------------------------------------ operations
def test_system_summary(client, target):
    s, _ = target
    d = _get(client, f"/api/systems/{s}/summary")
    _has(d, ["system", "risk", "risk_series", "common", "common_history", "coherent_shifts",
             "campaign_graph", "campaigns", "engine_health"])
    _has(d["campaign_graph"], ["nodes", "edges"])
    _no_nan(d)


def test_detectors_health(client, rt):
    d = _get(client, "/api/detectors/health")
    _has(d, ["engines", "detectors", "n_errors"])
    names = {e["name"] for e in d["engines"]}
    assert "behavior.risk" in names and "behavior.portrait" in names
    for e in d["engines"]:
        _has(e, ["name", "last_run", "last_error", "error_count", "last_error_ts", "series"])
        for srs in e["series"]:
            _has(srs, ["name", "last_write_ts", "staleness_s"])
    risk = next(e for e in d["engines"] if e["name"] == "behavior.risk")
    assert any(x["name"] == "behavior.risk" and x["staleness_s"] == 0 for x in risk["series"])
    for s, rows in d["detectors"].items():
        assert s in rt.store.systems()
        for row in rows:
            _has(row, ["detector", "family", "ks", "rate_ratio", "weight_mult",
                       "degraded_share"])
    _no_nan(d)


def test_store_memory(client):
    d = _get(client, "/api/store/memory")
    _has(d, ["report", "approx_bytes", "entities"])


def test_eval_report_404_then_served(client, tmp_path, monkeypatch):
    missing = tmp_path / "none.json"
    monkeypatch.setenv("APPMON_EVAL_REPORT", str(missing))
    from app.api import routes_v2
    monkeypatch.setattr(routes_v2, "_eval_report_paths", lambda: [str(missing)])
    resp = client.get("/api/eval/report")
    assert resp.status_code == 404 and "error" in resp.json()
    rep = {"summary": {"n_runs": 1, "all_pass": True}, "gates": {}, "scenarios": []}
    path = tmp_path / "eval_report.json"
    path.write_text(json.dumps(rep), encoding="utf-8")
    monkeypatch.setattr(routes_v2, "_eval_report_paths", lambda: [str(path)])
    d = _get(client, "/api/eval/report")
    assert d["summary"]["all_pass"] is True


def test_static_frontend_served(client):
    resp = client.get("/app/")
    assert resp.status_code == 200 and "<html" in resp.text.lower()
    for f in ("js/app.js", "js/api.js", "js/charts.js", "js/pages.js", "css/app.css"):
        assert client.get(f"/app/{f}").status_code == 200, f


# ------------------------------------------------------------------ frontend
@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_frontend_js_syntax():
    js_dir = os.path.join(ROOT, "frontend", "js")
    files = sorted(f for f in os.listdir(js_dir) if f.endswith(".js"))
    assert files
    for f in files:
        res = subprocess.run(["node", "--check", os.path.join(js_dir, f)],
                             capture_output=True, text=True, timeout=60)
        assert res.returncode == 0, f"{f}: {res.stderr}"
