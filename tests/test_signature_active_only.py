"""lib-4 signature engines evaluate only the entities that can match this tick
(round 7, evaluation-harness owner; progressive.md 16.14.12 item 12).

`signature.rule_match` used to take a store snapshot of EVERY known entity on
every tick and `signature.correlation` read the match index of every known
entity, so the per-tick cost grew with idle known IPs (3 -> 52 ms per tick
with 400 idle IPs; the bounded-mode idle-IP test was flaky because of it).
A signature that needs a passive raw or probe metric can only match an entity
with a record of this tick, and a composite rule only an entity with a
primitive match inside its window. These tests check (a) that the restricted
engines produce exactly the matches of the reference scan over every known
entity, on the production registry at 60 s and 900 s ticks with idle IPs,
probe-only servers and derived values written for idle IPs, and (b) by
operation counts (not wall time) that their work no longer grows with idle
known entities.
"""
from __future__ import annotations

import numpy as np
import pytest

from helpers import DT, T0, add_obs_tick, make_store, run_engine

from app.core.store import MetricStore
from app.engines.signature import correlation as CO
from app.engines.signature import rule_match as RM
from app.engines.signature.store import Signature, SignatureStore
from app.models.schema import AcquisitionMethod, DerivedMetric, Observation, Reachability
from app.pipeline import build
from app.pipeline.orchestrator import Pipeline

ACTIVE = [f"192.168.1.{i}" for i in range(1, 6)]


def _obs(ts, ip, r, **kw):
    base = dict(ts=ts, system="oa", entity=ip, peer="192.168.100.100", l3_proto="ip",
                l4_proto="tcp", dst_port=8080, bytes_up=int(r.integers(300, 3000)),
                bytes_down=int(r.integers(1000, 30000)), pkts_up=4, pkts_down=6,
                app_proto="http", http_method=str(r.choice(["GET", "POST"])), http_host="oa",
                http_path=f"/oa/page{int(r.integers(0, 5))}",
                http_status=int(r.choice([200, 200, 401, 500])), user_agent=f"ua-{ip}")
    base.update(kw)
    return Observation(**base)


def _probe(ts, server, lost):
    return Observation(ts=ts, system="oa", entity=server, method=AcquisitionMethod.ACTIVE_PROBE,
                       dst_port=443, rtt_ms=0.0 if lost else 3.0,
                       reachability=Reachability.TIMEOUT if lost else Reachability.REACHABLE)


def _run(scan_all: bool, dt: float, ticks: int, n_idle: int = 30):
    reg = build.build_registry()
    for e in reg._engines:
        if isinstance(e, (RM.RuleMatchEngine, CO.CorrelationEngine)):
            e.scan_all = scan_all
    st = MetricStore()
    pipe = Pipeline(st, reg, window_s=int(dt),
                    config={"strict": True, "tz": "Asia/Shanghai", "grain_mode": "canonical"})
    r = np.random.default_rng(3)
    for h in range(ticks):
        t1 = T0 + (h + 1) * dt
        o = []
        for ip in ACTIVE[: 2 + h % 4]:                      # activity varies by tick
            o += [_obs(t1 - dt + r.uniform(0, dt), ip, r) for _ in range(int(r.integers(1, 8)))]
        if h == 0:                                          # seen once, then idle
            o += [_obs(t1 - 5.0, f"10.9.0.{k}", r) for k in range(n_idle)]
        if h % 3 == 0:                                      # a probe-only server
            o += [_probe(t1 - 2.0, "192.168.200.1", lost=(h % 2 == 0)) for _ in range(4)]
        if h == ticks // 2:                                 # a burst that matches
            o += [_obs(t1 - 1.0 - k, ACTIVE[0], r, http_method="POST", http_status=401,
                       bytes_up=9_000_000) for k in range(60)]
        pipe.run_tick(o, now=t1, dt=dt)
    eng = next(e for e in reg._engines if isinstance(e, RM.RuleMatchEngine))
    ms = sorted((m.ts, m.entity, m.signature_id, m.confidence, tuple(m.matched_terms))
                for m in st.matches(limit=1 << 30))
    return ms, eng.n_snapshots


@pytest.mark.parametrize("dt,ticks", [(900.0, 14), (60.0, 30)])
def test_restricted_engines_equal_the_full_scan(dt, ticks):
    got, n_new = _run(False, dt, ticks)
    want, n_ref = _run(True, dt, ticks)
    assert got == want
    assert any(sid == "service_degraded" for _, _, sid, _, _ in got)   # probe-only server
    assert any(not sid.startswith("composite:") and e == ACTIVE[0] for _, e, sid, _, _ in got)
    assert n_new < 0.5 * n_ref, (n_new, n_ref)


def _lib():
    sigs = SignatureStore()
    sigs.signatures += [
        Signature(id="upload", label="upload", category="transfer", severity="high",
                  all=[{"metric": "l4.bytes_up", "op": "gt", "value": 3e6}]),
        Signature(id="probe_down", label="down", category="other",
                  any=[{"metric": "probe.loss_ratio", "op": "ge", "value": 0.5}]),
    ]
    return sigs


def _store(n_idle):
    st = make_store()
    for k in range(n_idle):
        add_obs_tick(st, "erp", f"10.9.0.{k}", T0, {"l4.bytes_up": 1e3})
    return st


@pytest.mark.parametrize("n_idle", [10, 400])
def test_rule_match_work_is_independent_of_idle_entities(n_idle):
    st, eng = _store(n_idle), RM.RuleMatchEngine(_lib())
    run_engine(eng, st, T0)
    assert eng.n_snapshots == n_idle                       # all observed at T0
    before = eng.n_snapshots
    for k in range(1, 6):
        now = T0 + k * DT
        add_obs_tick(st, "erp", "10.0.0.1", now, {"l4.bytes_up": 5e6})
        run_engine(eng, st, now)
    # 5 ticks x 1 active entity, whatever the number of idle known entities
    assert eng.n_snapshots - before == 5
    assert len(st.matches("erp", "10.0.0.1", limit=10)) == 5


def test_probe_only_entity_and_derived_only_signature_still_evaluated():
    st = _store(50)
    sigs = _lib()
    # a signature that needs only a derived value: not observation-gated, it
    # keeps the scan over every known entity (an idle IP can match it)
    sigs.signatures.append(Signature(id="slow_drift", label="drift", category="other",
                                     all=[{"metric": "derived.drift_x", "op": "gt", "value": 1}]))
    assert [RM.observation_gated(s) for s in sigs.signatures] == [True, True, False]
    eng = RM.RuleMatchEngine(sigs)
    now = T0 + DT
    # probe metrics are written touch=False (raw/active_probe): not active ...
    add_obs_tick(st, "erp", "srv", now, {"probe.loss_ratio": 1.0}, touch=False)
    st.add_observation(Observation(ts=now - 3.0, system="erp", entity="srv",
                                   method=AcquisitionMethod.ACTIVE_PROBE))   # ... but observed this tick
    st.add_derived(DerivedMetric(name="derived.drift_x", value=5.0, ts=now, system="erp", window_s=DT,
                                 entity="10.9.0.7"))
    run_engine(eng, st, now)
    got = sorted((m.entity, m.signature_id) for m in st.matches("erp", limit=100))
    assert got == [("10.9.0.7", "slow_drift"), ("srv", "probe_down")]


def test_correlation_reads_only_entities_with_recent_matches(monkeypatch):
    st = _store(200)
    rules = [{"id": "auth_then_up", "sequence": ["auth", "transfer"], "window_s": 600,
              "severity": "high"}]
    from app.models.schema import Severity, SignatureMatch
    for k, cat in enumerate(("auth", "transfer")):
        st.add_match(SignatureMatch(system="erp", entity="10.9.0.3", ts=T0 + 100 * k,
                                    signature_id=cat, label=cat, category=cat, confidence=0.9,
                                    matched_terms=[], severity=Severity.LOW, evidence={}))
    calls = []
    real = st.matches

    def spy(*a, **kw):
        calls.append(kw.get("entity"))
        return real(*a, **kw)

    monkeypatch.setattr(st, "matches", spy)
    eng = CO.CorrelationEngine(rules)
    assert run_engine(eng, st, T0 + 200) == 1
    # one system-wide window read + one read of the matched entity, not 200
    assert calls == [None, "10.9.0.3"]
    ref = CO.CorrelationEngine(rules, scan_all=True)
    monkeypatch.setattr(st, "matches", real)
    assert run_engine(ref, st, T0 + 200 + 600) == 0          # outside the window
