"""B14 spec unit test (e): daily creep test (Mann-Kendall + Sen slope against
the class median, relative to the golden anchor) and its training-mode rule.

(e) An entity ramping +0.2 sigma per day on its volume group while its class
    stays flat for 14 days raises baseline_creep (p < 0.01 and |Sen slope| >
    0.05 log-units per day; feature.vec = 0.5 zr, so the slope is 0.1 log/day).
Training: the same scenario with ctx.training learns and scores (the creep
    accumulator latches) but emits no event.
Hourly cadence (3600 s) keeps 15 days x 3 entities under ~2 s per run.
"""
from __future__ import annotations

import numpy as np
import pytest

from helpers import T0, make_store, put_model, run_engine, set_trust

from app.engines.behavior.changepoint import ChangepointEngine
from app.engines.behavior.lib import emit, m_cp
from app.engines.behavior.lib.classkeys import ORG
from app.engines.behavior.lib.features import FEATURE_DIM, GROUPS

S = "erp"
DT = 3600.0
ENTS = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]            # 10.0.0.1 ramps
VOL = GROUPS["volume"]


def class_model():
    return {"assign": {f"{S}|{e}": {"role": "r1", "prob": 0.9} for e in ENTS},
            "roles": {"r1": {"name": "clerks", "members": [f"{S}|{e}" for e in ENTS]}},
            "version": 1}


def run_scenario(training: bool, days: int = 15, slope: float = 0.2, control=None):
    store, eng = make_store(), ChangepointEngine()
    put_model(store, *ORG, "model.class", class_model())
    if control is not None:
        put_model(store, S, ENTS[0], "model.control", control)
    rng = np.random.default_rng(17)
    n = int(days * 86400 / DT)
    for i in range(n):
        t = T0 + i * DT
        for j, e in enumerate(ENTS):
            zr = rng.standard_normal(FEATURE_DIM)
            if j == 0:
                zr[VOL] += slope * (t - T0) / 86400.0
            w = int(DT)
            store.add_vec(S, e, "behavior.zr", t, zr.astype(np.float32), window_s=w)
            store.add_vec(S, e, "feature.vec", t, (3.0 + 0.5 * zr).astype(np.float32), window_s=w)
            store.add_vec(S, e, "feature.active", t, np.ones(1, np.float32), window_s=w)
            if not training:
                set_trust(store, S, e, [t], 1.0)
            store.register_entity(S, e)
        run_engine(eng, store, t, dt=DT, training=training)
    return store, t


@pytest.fixture(scope="module")
def live():
    return run_scenario(training=False)


def test_e_ramp_raises_baseline_creep(live):
    store, t_end = live
    ev = store.events(system=S, kinds=["baseline_creep"])
    assert [x.entity for x in ev] == [ENTS[0]]          # once, only for the ramping entity
    x = ev[0]
    assert "volume" in x.axes
    assert x.p_value < 0.01 and x.p_by_detector["creep"] == x.p_value
    g = x.extra["groups"]["volume"]
    assert g["alarm"] and g["slope_log"] > 0.05
    assert g["slope_log"] == pytest.approx(0.1, rel=0.35)        # 0.2 sigma/day x 0.5 log/sigma
    assert x.window[0] < x.ts and x.dedupe_key.startswith("baseline_creep|")
    # the creep accumulator is latched with its axes; cp's descriptor reports it
    a = emit.read_dict(store, S, ENTS[0], emit.ACC_ALARM, t_end)
    assert a["creep"] == 1
    assert "volume" in emit.read_dict(store, S, ENTS[0], emit.AXES, t_end)["creep"]
    assert emit.read_row(store, S, ENTS[0], emit.PM, t_end)["creep"] < 0.1
    assert m_cp.descriptor(store, S, ENTS[0])["creep"]["volume"]["alarm"]
    # the flat peers: tested, no creep
    for e in ENTS[1:]:
        assert emit.read_dict(store, S, e, emit.ACC_ALARM, t_end)["creep"] == 0
        assert not m_cp.descriptor(store, S, e)["creep"]["volume"]["alarm"]


def test_training_learns_but_emits_no_event():
    store, t_end = run_scenario(training=True)
    assert store.events(system=S) == []
    assert emit.read_dict(store, S, ENTS[0], emit.ACC_ALARM, t_end)["creep"] == 1
    m = store.get_model(S, ENTS[0], "model.cp")
    assert m["learn"]["n"] > 0                          # learning ran (trust 1 in training)


def test_allow_drift_raises_the_slope_bar():
    """model.control.allow_drift (an accepted legitimate ramp, in log-units per
    day) is added to the 0.05 bar: a 0.1 log/day ramp under allow_drift 0.2
    is not creep."""
    store, _ = run_scenario(training=False, days=13, control={"allow_drift": 0.2})
    assert store.events(system=S, kinds=["baseline_creep"]) == []
