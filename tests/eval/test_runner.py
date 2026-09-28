"""Runner smoke on a tiny fake pack, fake generator and fake registry, so it
does not depend on the (unfinished) production engines: phases and clock,
training flags, per-tick collection, incident history, strict-mode
exception capture, stale-series detection, ablation, time budget,
pickling / process-pool execution and end-to-end scoring."""
import pickle
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np
import pytest

from app.core.engine import Engine, Registry
from app.engines.behavior.lib.detectors import DETECTORS
from app.engines.behavior.lib.emit import write_pvalues, write_scores
from app.eval.metrics import score_run
from app.eval.report import build_report
from app.eval.runner import SimulatedAnalyst, run_pack
from app.models.schema import (BehaviorEvent, DerivedMetric, Incident, MetricKind, Observation,
                               RawMetric, Severity)

S = "sys"
ATTACKED = "10.0.0.2"
ENTITIES = ["10.0.0.1", ATTACKED, "10.0.0.3"]
VT0 = 1_700_000_000.0
WARM = (4, 3600.0, True)
SCEN = (8, 900.0, False)
T_START = VT0 + WARM[0] * WARM[1] + 3 * SCEN[1]       # the 3rd scenario tick
T_END = T_START + 3 * SCEN[1]


@dataclass
class FakePack:
    name: str = "fake"
    tz: str = "Asia/Shanghai"
    calendar: dict = field(default_factory=lambda: {"holidays": [], "makeup_workdays": []})
    phases: List[Any] = field(default_factory=lambda: [WARM, SCEN])
    scenarios: List[str] = field(default_factory=lambda: ["T1"])
    seeds: List[int] = field(default_factory=lambda: [0])


@dataclass
class P:
    entity: str
    archetype: str


class FakeGen:
    def __init__(self, seed: int = 0, pack: Any = None) -> None:
        self.vt = VT0
        self.seed = seed
        self.calls: List[tuple] = []
        self.systems = {S: [P(e, "interactive") for e in ENTITIES]}
        self.truth: List[dict] = []

    def step(self, dt: float, live: bool = False) -> List[Observation]:
        self.vt += dt
        self.calls.append((dt, live))
        if live and not self.truth:
            self.truth.append({"scenario_id": "T1", "pack": "fake", "system": S,
                               "entities": [ATTACKED], "t_start": T_START, "t_end": T_END,
                               "label": "malicious", "expected_detectors": ["B04 marg_int"],
                               "expected_axes": ["volume"], "max_ttd": 1800,
                               "required_severity": "high", "perturbed_features": ["bytes_up"]})
        return [Observation(ts=self.vt - 1.0, system=S, entity=e, bytes_up=100) for e in ENTITIES]


class FakeRaw(Engine):
    name = "fake_raw"
    layer = "raw"

    def run(self, ctx, observations=None):
        for o in observations or []:
            ctx.store.add_raw(RawMetric(name="l4.flows", value=1.0, ts=ctx.now, system=o.system,
                                        entity=o.entity, kind=MetricKind.COUNTER))
        return len(observations or [])


class FakeScorer(Engine):
    """Writes calibrated p-values, e_day, risk and a heartbeat vector for
    every entity; after `stop_after` live ticks it stops one vector series
    (stale) while the others keep going."""

    name = "fake_scorer"
    layer = "behavior"

    def __init__(self, stop_after: int = 10 ** 9, **kw):
        super().__init__(**kw)
        self.stop_after = stop_after
        self.live = 0

    def run(self, ctx, observations=None):
        st = ctx.store
        self.live += 0 if ctx.training else 1
        for e in st.entities(S):
            u = (zlib.crc32(f"{e}|{ctx.now}".encode()) % 10_000 + 0.5) / 10_000
            p = {d: (u + i / len(DETECTORS)) % 1.0 for i, d in enumerate(DETECTORS)}
            attacked = e == ATTACKED and ctx.now >= T_START and not ctx.training
            if attacked:
                p["marg_int"] = 1e-7
            write_scores(st, S, e, ctx.now, {"marg_int": 1.0}, axes={"marg_int": ["volume"]},
                         acc_alarm={"cusum": 0})
            write_pvalues(st, S, e, ctx.now, p)
            st.add_derived(DerivedMetric("behavior.e_day", float(p["marg_int"]) * 86400 / ctx.dt,
                                         ctx.now, S, e, int(ctx.dt)))
            st.add_derived(DerivedMetric("behavior.risk", {"score": 70.0 if attacked else 5.0},
                                         ctx.now, S, e, int(ctx.dt)))
            st.add_derived(DerivedMetric("behavior.p_family", {"intensity": 0.2}, ctx.now, S, e,
                                         int(ctx.dt)))
            st.add_vec(S, e, "behavior.heartbeat", ctx.now, [1.0])
            if self.live <= self.stop_after:
                st.add_vec(S, e, "behavior.fragile", ctx.now, [1.0])
        return 1


class FakeIncident(Engine):
    """Opens a LOW incident on the attacked entity at onset, escalates it to
    HIGH one tick later and closes it after the attack."""

    name = "fake_incident"
    layer = "behavior"

    def run(self, ctx, observations=None):
        if ctx.training:
            return 0
        st = ctx.store
        cur = st.incidents(system=S, entity=ATTACKED)
        if ctx.now >= T_START and not cur:
            inc = Incident(system=S, entity=ATTACKED, axes=["volume"], opened=ctx.now,
                           last_seen=ctx.now, severity=Severity.LOW,
                           evidence=[{"ts": ctx.now, "p_by_detector": {"marg_int": 1e-7}}])
            iid = st.put_incident(inc)
            st.add_event(BehaviorEvent(system=S, entity=ATTACKED, ts=ctx.now, kind="incident",
                                       score=1.0, severity=Severity.LOW, incident_id=iid,
                                       extra={"state": "open"}))
        elif cur and cur[0].severity == Severity.LOW:
            inc = cur[0]
            inc.severity, inc.last_seen = Severity.HIGH, ctx.now
            st.put_incident(inc)
            st.add_event(BehaviorEvent(system=S, entity=ATTACKED, ts=ctx.now, kind="incident",
                                       score=1.0, severity=Severity.HIGH, incident_id=inc.id,
                                       extra={"state": "escalate"}))
        elif cur and ctx.now > T_END and cur[0].status != "closed":
            inc = cur[0]
            inc.status, inc.close_reason, inc.last_seen = "closed", "returned", ctx.now
            st.put_incident(inc)
        return 1


class Boom(Engine):
    name = "boom"
    layer = "behavior"

    def run(self, ctx, observations=None):
        if not ctx.training and ctx.now >= T_START:
            raise RuntimeError("kaboom")
        return 0


def fake_registry(pack=None, config=None):
    reg = Registry()
    reg.add(FakeRaw(), FakeScorer(), FakeIncident())
    return reg


def boom_registry():
    reg = fake_registry()
    reg.add(Boom())
    return reg


def stale_registry():
    reg = Registry()
    reg.add(FakeRaw(), FakeScorer(stop_after=2))
    return reg


def _run(**kw):
    kw.setdefault("registry_factory", fake_registry)
    return run_pack(FakePack(), 0, generator_factory=FakeGen, **kw)


def test_phases_clock_and_collection():
    res = _run()
    assert res.aborted is None and not res.exceptions and not res.stale
    assert res.config["strict"] is True and res.config["tz"] == "Asia/Shanghai"
    assert [p["training"] for p in res.phases] == [True, False]
    # scenario ticks are stamped with gen.vt and the phase dt
    expect = VT0 + WARM[0] * WARM[1] + SCEN[1] * np.arange(1, SCEN[0] + 1)
    np.testing.assert_allclose(res.tick_ts, expect)
    assert res.scenario_dt == 900.0 and res.scenario_window == (expect[0] - 900.0, expect[-1])
    ticks = res.timings["ticks"]
    assert ticks.shape == (WARM[0] + SCEN[0], len(res.timings["columns"]))
    assert list(ticks[:, 3]) == [1] * WARM[0] + [0] * SCEN[0]
    assert res.timings["engine_ms"].shape == (12, 3)
    ser = res.series[f"{S}|{ATTACKED}"]
    assert ser["p"].shape == (SCEN[0], len(DETECTORS)) and ser["ts"].size == SCEN[0]
    assert np.isfinite(ser["e_day"]).all() and ser["risk"].max() == 70.0
    assert ser["p_family"][0, 0] == pytest.approx(0.2)
    assert res.personas[f"{S}|{ATTACKED}"]["archetype"] == "interactive"
    assert res.truth and res.truth[0]["scenario_id"] == "T1"
    assert set(res.entity_first_seen) == {f"{S}|{e}" for e in ENTITIES}


def test_incident_history_and_scoring_end_to_end():
    res = _run()
    (inc,) = res.incidents
    sev = [(h["ts"], h["severity"], h["status"]) for h in inc["history"]]
    assert sev[0] == (T_START, "low", "open") and sev[1] == (T_START + 900.0, "high", "open")
    assert sev[-1][2] == "closed"
    assert [e["extra"]["state"] for e in res.events if e["kind"] == "incident"] == ["open",
                                                                                    "escalate"]
    snaps = res.models["baseline_snaps"]["T1"]
    assert snaps["pre"]["ts"] == T_START - 900.0 and "post" in snaps
    sc = score_run(res)
    (o,) = sc["scenarios"]
    assert o["detected"] and o["ttd_ticks"] == 2 and o["within_deadline"]
    assert o["notifications"] == 2 and o["matched"]
    assert sc["far"]["n_control"] == 2 and sc["far"]["n_low"] == 0
    assert sc["calibration"]["ks"]["marg_int"]["n"] == 2 * SCEN[0]
    rep = build_report([sc])
    assert rep["gates"]["15_robustness"]["pass"] is True
    assert rep["gates"]["1_detection"]["value"] == 1.0


def test_strict_exceptions_recorded_and_abort_mode():
    res = _run(registry_factory=boom_registry)
    assert res.aborted is None                       # 'record' keeps the run going
    assert len(res.exceptions) == 6 and res.exceptions[0]["engine"] == "boom"
    assert "kaboom" in res.exceptions[0]["error"]
    assert res.incidents                             # later engines still ran
    assert score_run(res)["robustness"]["exceptions"] == 6
    res = _run(registry_factory=boom_registry, on_error="abort")
    assert res.aborted.startswith("engine_error") and len(res.exceptions) == 1
    with pytest.raises(RuntimeError):
        _run(registry_factory=boom_registry, on_error="raise")


def test_stale_series_detected():
    res = _run(registry_factory=stale_registry)
    names = {(s["entity"], s["name"]) for s in res.stale}
    assert names == {(e, "behavior.fragile") for e in ENTITIES}


def test_disable_engine_and_time_budget():
    res = _run(disable_engines=["fake_incident"])
    assert res.disabled_engines == ["fake_incident"] and not res.incidents
    assert score_run(res)["scenarios"][0]["detected"] is False
    res = _run(time_budget_s=0.0)
    assert res.aborted == "time_budget" and res.timings["ticks"].shape[0] == 1


def test_simulated_analyst_labels():
    an = SimulatedAnalyst(seed=0, noise=0.0, per_day=96.0)   # one label per 900-s tick
    res = _run(analyst=an, keep_store=True)
    assert res.labels_added == 1 and an.n == 1
    (lb,) = res.store.labels()
    assert lb.verdict == "tp" and lb.target_type == "incident" and lb.scope == "this"


def test_simulated_analyst_dismisses_fp_as_a_pattern():
    """fp verdicts are labelled with scope 'pattern' (B23 turns only widened
    scopes into suppression policies, which gate 12 measures)."""
    an = SimulatedAnalyst(seed=0, noise=1.0, per_day=96.0)   # every verdict flipped: tp -> fp
    res = _run(analyst=an, keep_store=True)
    (lb,) = res.store.labels()
    assert lb.verdict == "fp" and lb.scope == "pattern"


def _pool_job(seed):
    res = run_pack(FakePack(), seed, registry_factory=fake_registry, generator_factory=FakeGen,
                   keep_store=True)
    blob = pickle.dumps(res)
    back = pickle.loads(blob)
    assert back.store is None
    return score_run(back)["scenarios"][0]["detected"]


def test_pickle_and_process_pool():
    assert _pool_job(0) is True
    with ProcessPoolExecutor(max_workers=2) as ex:
        assert list(ex.map(_pool_job, [0, 1])) == [True, True]
