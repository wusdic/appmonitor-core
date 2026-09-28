"""End to end: the full v2 registry (build.build_registry) on the 'mini' eval
pack in strict mode (docs/lib3/integration.md).

The run is the eval harness's own (eval/runner.run_pack): 72 x 3600 s
warm-up with training=True, then 128 x 900 s live ticks stamped with the
generator's virtual clock, ctx.config['strict'] = True so any engine exception
is re-raised and recorded. ~25 s on one core.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.engines.behavior.lib import m_baseline as MB
from app.eval.runner import run_pack
from app.pipeline import build

V2_BEHAVIOR_ORDER = [
    "behavior.feature_vector", "behavior.peer_group", "behavior.baseline",
    "behavior.likelihood", "behavior.common_mode", "behavior.multivariate",
    "behavior.rhythm", "behavior.novelty", "behavior.client_identity", "behavior.sequence",
    "behavior.timing", "behavior.beacon", "behavior.budget", "behavior.changepoint",
    "behavior.identity_model", "behavior.attribution", "behavior.entity_link",
    "behavior.class_monitor", "behavior.cross_system", "behavior.feedback", "behavior.calibration", "behavior.fusion",
    "behavior.risk", "behavior.incident", "behavior.governor", "behavior.explain",
    "behavior.portrait",
]
V1_ENGINES = {"behavior.anomaly", "behavior.drift", "behavior.fingerprint",
              "behavior.clustering"}


@pytest.fixture(scope="module")
def run():
    res = run_pack("mini", seed=0, strict=True, on_error="record", keep_store=True)
    yield res
    res.store = None


def _real_keys(res):
    return sorted(k for k in res.series if "|class:" not in k)


def test_registry_is_the_full_v2_set_in_contract_order():
    reg = build.build_registry()
    names = [e.name for e in reg.ordered()]
    assert [n for n in names if n.startswith("behavior.")] == V2_BEHAVIOR_ORDER
    assert names[:8] == ["raw.l2l3", "raw.l4flow", "raw.http", "raw.tls", "raw.dns",
                         "raw.active_probe", "raw.action_token", "raw.client_stack"]
    assert names[8:15] == ["derived.aggregation", "derived.periodicity", "derived.trend",
                           "derived.ratio", "derived.entropy", "derived.graph",
                           "derived.session"]
    assert names[-2:] == ["signature.rule_match", "signature.correlation"]
    assert not V1_ENGINES & set(names)
    # p2=False leaves out the P2 engines (B21 cross_system) and nothing else
    names0 = [e.name for e in build.build_registry(p2=False).ordered()]
    assert names0 == [n for n in names if n != "behavior.cross_system"]


def test_strict_run_has_no_engine_errors(run):
    assert run.aborted is None
    assert run.exceptions == []
    assert len(run.health) >= 42                 # every registered engine reported health
    assert all(v.get("runs", 0) > 0 for v in run.health.values() if isinstance(v, dict))
    bad = {k: v for k, v in run.health.items()
           if isinstance(v, dict) and int(v.get("error_count", 0) or 0) > 0}
    assert bad == {}
    assert run.stale == []                       # no lib-3 series stopped while its entity ran
    assert len(run.tick_ts) == 128


def test_feature_score_p_risk_series_are_populated(run):
    st = run.store
    keys = _real_keys(run)
    assert len(keys) == 8
    for k in keys:
        s, e = k.split("|", 1)
        ts, vec = st.vec_since(s, e, "feature.vec", run.scenario_window[0])
        assert len(ts) > 0 and np.isfinite(vec).any(), k
        ts, sc = st.vec_since(s, e, "behavior.score", run.scenario_window[0])
        assert len(ts) > 0 and np.isfinite(sc).any(), k
        ser = run.series[k]
        assert np.isfinite(ser["p"]).any(), k              # calibrated p (B24)
        assert np.isfinite(ser["e_day"]).any(), k           # fusion (B25)
        assert np.isfinite(ser["risk"]).all(), k            # risk every tick (B26)
        # the baseline kept learning: a rollback never erases warm-up rows
        m = st.get_model(s, e, MB.MODEL)
        assert MB.n_eff(m) > 0.0, k


def _threats(res):
    """{'sys|ip': (t_start, t_end)} of the malicious scenarios."""
    out = {}
    for row in res.truth:
        if row.get("label") != "malicious":
            continue
        s = row.get("system")
        for e in row.get("entities") or []:
            key = e if "|" in e else f"{s}|{e}"
            out[key] = (float(row.get("t_start", -math.inf)), float(row.get("t_end", math.inf)))
    return out


def test_an_incident_opens_on_a_threat_entity(run):
    threat = _threats(run)
    assert threat
    hits = [i for i in run.incidents
            if f"{i['system']}|{i['entity']}" in threat
            or set(threat) & {f"{i['system']}|{x}" for x in i.get("entities") or []}]
    assert hits, "no incident on any malicious scenario entity"
    t0, t1 = run.scenario_window
    assert any(t0 <= float(i["opened"]) <= t1 for i in hits)
    assert all(not math.isnan(float(i.get("risk") or 0.0)) for i in hits)
    # the pipeline keeps feeding the incident while the threat is active:
    # alarm / finding / risk evidence stamped inside the scenario's own window
    live = []
    for i in hits:
        w = threat.get(f"{i['system']}|{i['entity']}")
        if w is None:
            continue
        live += [ev for ev in i.get("evidence") or []
                 if ev.get("source") in ("alarm", "event", "risk")
                 and w[0] <= float(ev.get("ts", -math.inf)) <= w[1]]
    assert live, "no incident evidence inside a threat's scenario window"


def test_timings_are_recorded_per_engine(run):
    tm = run.timings
    E = np.asarray(tm["engine_ms"])
    assert E.shape == (72 + 128, len(tm["engine_names"]))
    assert np.all(np.isfinite(E)) and np.all(E >= 0.0)
    behavior = [i for i, n in enumerate(tm["engine_names"]) if n.startswith("behavior.")]
    assert E[:, behavior].sum(axis=1).max() > 0.0


def test_runtime_warmup_plan():
    """spec v2.1 (cadence.md §10): the Runtime warms up 120 x 3600 s + 192 x
    900 s by default, `warmup_ticks=n` keeps the v2 plan n x 900 s, and a
    warning names a last warm-up phase that misses a day type."""
    import datetime as dt

    from app.pipeline.generator import Clock
    rt = build.Runtime(live_period_s=0.0)
    assert rt.warmup_plan == [(120, 3600.0), (192, 900.0)]
    assert rt.warmup_ticks == 312 and rt.warmup_span_s() == 7 * 86400.0
    assert "120 x 3600 s + 192 x 900 s" in rt.plan_text()
    assert rt.config["grain_mode"] == "canonical"
    clk = Clock(rt.gen.clock.tz)
    sun = clk.epoch(dt.date(2025, 3, 9), 0.0)        # Fri + Sat before: both day types
    mon = clk.epoch(dt.date(2025, 3, 10), 0.0)       # Sat + Sun before: no workday
    assert rt.plan_warnings(sun) == []
    assert rt.plan_warnings(mon) and "workday" in rt.plan_warnings(mon)[0]
    legacy = build.Runtime(warmup_ticks=5, live_period_s=0.0)
    assert legacy.warmup_plan == [(5, 900.0)] and legacy.warmup_ticks == 5
    custom = build.Runtime(warmup_plan=[(2, 3600.0), (4, 900.0)], live_period_s=0.0)
    custom.warmup()
    assert custom.pipeline.tick_count == 6
