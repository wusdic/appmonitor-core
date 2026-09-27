"""B25 FusionEngine: docs/lib3/engines.md B25 unit tests (a), (d), (e), (f),
the decision paths, the reading rules, BH and the edge cases.

The statistical checks (b) and (c) live in test_b25_fusion_stats.py and the
meta-ring gating / cadence tests in test_b25_fusion_gating.py (each file
stays under the 5 s budget).

`Rig` plays the surrounding pipeline of one tick: B24 writes behavior.p
(emit.write_pvalues), the detectors write axes / acc_alarm / degraded
(emit.write_scores), B01 writes feature.tctx, B25 runs, then B28 writes trust
for that tick. With no trusted history the meta rings are empty, so q = the
raw wHMP p (small-sample blend with weight 0): tests set p from the e_day
they want, p = e_day * dt / 86400.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, Optional

import numpy as np
import pytest

from helpers import DT, T0, make_store, make_tctx, put_model, run_engine, set_trust

from app.engines.behavior import fusion as F
from app.engines.behavior.fusion import FusionEngine
from app.engines.behavior.lib import combine, emit, m_calib
from app.engines.behavior.lib.detectors import DETECTOR_INDEX, N_DETECTORS
from app.models.schema import BehaviorEvent, DerivedMetric, MetricKind, Severity, SignatureMatch

S, E = "sys", "10.0.0.1"
NAN = float("nan")


def p_at(e_day: float, dt: float = DT) -> float:
    return e_day * dt / 86400.0


class Rig:
    """One simulated pipeline around B25 (see module docstring)."""

    def __init__(self, entities: Iterable[str] = (E,), dt: float = DT, t0: float = T0) -> None:
        self.store = make_store()
        self.eng = FusionEngine()
        self.dt = dt
        self.t = t0
        self.keys = list(entities)
        for e in self.keys:
            self.store.register_entity(S, e)

    def step(self, pvals: Dict[str, Dict[str, float]], axes: Optional[Dict] = None,
             acc: Optional[Dict] = None, degraded: Optional[Dict] = None,
             trust: Optional[float] = 1.0, quarantine: float = 0.0, dt: Optional[float] = None,
             training: bool = False, config: Optional[Dict] = None) -> float:
        dt = self.dt if dt is None else dt
        ts = self.t
        for e, pv in pvals.items():
            emit.write_pvalues(self.store, S, e, ts, pv)
            if axes or acc or degraded:
                emit.write_scores(self.store, S, e, ts, {}, axes=(axes or {}).get(e),
                                  acc_alarm=(acc or {}).get(e),
                                  degraded=(degraded or {}).get(e))
            if not e.startswith("class:"):
                t = make_tctx(ts, dt=dt)
                t.pop("daypart_id", None)
                self.store.add_derived(DerivedMetric(name="feature.tctx", value=t, ts=ts,
                                                     system=S, entity=e, window_s=int(dt),
                                                     kind=MetricKind.CATEGORICAL))
        self.n = run_engine(self.eng, self.store, ts, training=training, dt=dt, config=config)
        if trust is not None:
            for e in self.keys:
                set_trust(self.store, S, e, [ts], trust, quarantine=quarantine)
        self.t = ts + dt
        self.dt = dt
        return ts

    def alarm(self, e: str = E, ts: Optional[float] = None) -> Optional[Dict]:
        ts = self.t - self.dt if ts is None else ts
        m = self.store.latest_derived(S, e, F.ALARM)
        return dict(m.value) if m is not None and m.ts == ts else None

    def v(self, name: str, e: str = E, ts: Optional[float] = None) -> float:
        ts = self.t - self.dt if ts is None else ts
        row = self.store.vec_at(S, e, name, ts)
        return NAN if row is None else float(row[0])

    def pf(self, e: str = E, ts: Optional[float] = None) -> Optional[Dict]:
        ts = self.t - self.dt if ts is None else ts
        m = self.store.latest_derived(S, e, F.P_FAMILY)
        return dict(m.value) if m is not None and m.ts == ts else None


def sev(rig: Rig, e: str = E) -> Optional[str]:
    a = rig.alarm(e)
    return None if a is None else a["severity"]


# ======================================================================= spec
def test_engine_contract():
    eng = FusionEngine()
    assert eng.name == "behavior.fusion" and eng.layer == "behavior" and eng.interval == 1
    for name in ("behavior.p_family", "behavior.q_inst", "behavior.q_all", "behavior.e_day",
                 "behavior.evidence", "behavior.alarm"):
        assert name in eng.produces
    for name in ("behavior.p", "behavior.axes", "behavior.acc_alarm", "model.feedback",
                 "behavior.calib_health", "behavior.common.flag", "feature.tctx"):
        assert name in eng.consumes


def test_a_masking_regression():
    """(a) [1e-3, 0.999, 0.5] in one family -> p_family ~ 3e-3 (ACAT would give 0.5)."""
    row = np.full(N_DETECTORS, np.nan)
    for d, p in (("marg_int", 1e-3), ("t2", 0.999), ("budget_vol", 0.5)):
        row[DETECTOR_INDEX[d]] = p
    fz = F.fuse(row)
    assert fz.p_family == {"intensity": pytest.approx(3.0 / (1e3 + 1 / 0.999 + 2.0), rel=1e-9)}
    assert fz.p_family["intensity"] == pytest.approx(3e-3, rel=0.01)
    assert fz.p_all == pytest.approx(fz.p_family["intensity"])
    assert combine.whmp([1e-3, 0.999, 0.5]) == pytest.approx(fz.p_family["intensity"])
    # through the engine: behavior.p_family is written with the same value
    rig = Rig()
    rig.step({E: {"marg_int": 1e-3, "t2": 0.999, "budget_vol": 0.5}})
    assert rig.pf()["intensity"] == pytest.approx(3e-3, rel=0.01)


def test_fuse_instantaneous_vs_all_and_weights():
    row = np.full(N_DETECTORS, np.nan)
    row[DETECTOR_INDEX["marg_int"]] = 0.2       # intensity, inst
    row[DETECTOR_INDEX["budget_vol"]] = 1e-4    # intensity, acc
    row[DETECTOR_INDEX["cusum"]] = 1e-3         # change, acc only
    row[DETECTOR_INDEX["novelty"]] = 0.5        # categorical, inst
    fz = F.fuse(row)
    assert set(fz.p_family) == {"intensity", "change", "categorical"}
    # p_inst: instantaneous members only, of the families that have them
    assert fz.p_inst == pytest.approx(combine.whmp([0.2, 0.5]))
    assert fz.p_all == pytest.approx(combine.whmp(
        [combine.whmp([0.2, 1e-4]), 1e-3, 0.5]))
    # family weights (feedback) and health weights (weight_mult)
    fw = {f: 1.0 for f in fz.p_family}
    fw["change"] = 0.1
    fz2 = F.fuse(row, fw)
    assert fz2.p_all == pytest.approx(combine.whmp(
        [fz.p_family["intensity"], 1e-3, 0.5], [1.0, 0.1, 1.0]))
    wm = [1.0] * N_DETECTORS
    wm[DETECTOR_INDEX["budget_vol"]] = 0.5
    fz3 = F.fuse(row, None, wm)
    assert fz3.p_family["intensity"] == pytest.approx(combine.whmp([0.2, 1e-4], [1.0, 0.5]))
    assert fz3.w_family["intensity"] == pytest.approx(0.75)


def test_fuse_nan_never_p_one():
    row = np.full(N_DETECTORS, np.nan)
    fz = F.fuse(row)
    assert fz.p_family == {} and math.isnan(fz.p_all) and math.isnan(fz.p_inst)
    row[DETECTOR_INDEX["marg_int"]] = 1e-3
    row[DETECTOR_INDEX["t2"]] = NAN
    row[DETECTOR_INDEX["cusum"]] = 0.3
    fz = F.fuse(row)
    assert fz.p_family["intensity"] == pytest.approx(1e-3)
    assert fz.p_inst == pytest.approx(1e-3)        # change has no inst member
    assert F.meta_score(NAN) != F.meta_score(NAN)  # NaN stays NaN
    assert math.isnan(F.meta_q(None, NAN, 0.5, NAN))


def test_evidence_thresholds():
    assert F.evidence_h(900) == pytest.approx(5.31, abs=0.01)
    assert F.evidence_h(60) == pytest.approx(8.19, abs=0.01)
    assert F.evidence_h(3600) == pytest.approx(3.83, abs=0.01)
    assert F.evidence_h(900, 1.2) == pytest.approx(1.2 * F.evidence_h(900))


@pytest.mark.parametrize("dt,first", [(900.0, 4), (60.0, 6)])
def test_d_persistent_q_alarms_on_time(dt, first):
    """(d) persistent q = 0.01 alarms at tick 4 (900 s) / tick 6 (60 s), not before."""
    h = F.evidence_h(dt)
    S_ = 0.0
    fired = []
    for k in range(1, 10):
        S_, upd, al = F.evidence_update(S_, 0.01, h)
        assert upd
        fired.append(al)
    assert fired.index(True) + 1 == first
    # same through the engine (q_inst = the single instantaneous p)
    rig = Rig(dt=dt)
    for k in range(1, first + 1):
        rig.step({E: {"novelty": 0.01}}, trust=None)
        a = rig.alarm()
        assert rig.v(F.Q_INST) == pytest.approx(0.01, rel=1e-5)
        if k < first:
            assert a is None
        else:
            assert a is not None and a["path"] == F.PATH_EVIDENCE and a["onset"] is True
            assert a["severity"] == "low" and a["h"] == pytest.approx(h)


def test_d_isolated_q_does_not_alarm():
    rig = Rig()
    for q in (0.5, 0.3, 0.005, 0.6, 0.4, 0.7):
        rig.step({E: {"novelty": q}}, trust=None)
        assert rig.alarm() is None
    S_, _, al = F.evidence_update(0.0, 0.005, F.evidence_h(900))
    assert not al and S_ == pytest.approx(-math.log(0.005) - 3.0)


def test_evidence_nan_leaves_state():
    S_, upd, al = F.evidence_update(4.0, NAN, 1.0)
    assert S_ == 4.0 and not upd and not al
    S_, upd, al = F.evidence_update(NAN, 0.5, 5.0)
    assert S_ == 0.0 and upd                        # corrupt state restarts
    rng = np.random.default_rng(3)
    q = rng.random(500)
    q[::17] = NAN
    q[100:110] = 1e-4
    h = F.evidence_h(900)
    path, onsets = F.evidence_path(q, h, chunk=64)
    s, n_on, prev = 0.0, 0, 0.0
    for i, x in enumerate(q):
        s, upd, al = F.evidence_update(s, x, h)
        assert path[i] == pytest.approx(s, abs=1e-9)
        n_on += int(upd and al and prev < h)
        prev = s
    assert n_on == onsets >= 1


def test_e_volume_only_medium_then_categorical_allows_high():
    """(e) volume-only at e_day 1e-6 -> MEDIUM; + categorical at 0.01 -> HIGH."""
    rig = Rig()
    rig.step({E: {"marg_int": p_at(1e-6)}})
    a = rig.alarm()
    assert a["severity"] == "medium" and a["axes"] == ["volume"]
    assert a["e_day"] == pytest.approx(1e-6, rel=1e-3)
    # still MEDIUM once HIGH is corroborated (2 consecutive ticks): the volume cap
    rig.step({E: {"marg_int": p_at(1e-6)}})
    a = rig.alarm()
    assert a["severity"] == "medium" and "volume_only" in a["rules"]

    rig = Rig()
    rig.step({E: {"marg_int": p_at(1e-6), "novelty": p_at(0.01)}})
    a = rig.alarm()
    assert a["severity"] == "high"
    assert a["axes"] == ["categorical", "volume"] and a["n_axes"] == 2
    assert set(a["families"]) == {"intensity", "categorical"}


def test_f_single_family_single_tick_is_medium():
    """(f) one family at e_day 1e-5 for a single tick, no corroboration -> MEDIUM."""
    rig = Rig()
    rig.step({E: {"novelty": p_at(1e-5)}})
    a = rig.alarm()
    assert a["severity"] == "medium" and "corroboration_medium" in a["rules"]
    assert a["path"] == F.PATH_SINGLE
    # the next tick is back to normal: no alarm, and no late escalation
    rig.step({E: {"novelty": 0.5}})
    assert rig.alarm() is None or rig.alarm()["severity"] in ("low", "medium")


# ================================================================ corroboration
def test_high_by_two_consecutive_ticks():
    rig = Rig()
    rig.step({E: {"novelty": p_at(1e-4)}})
    assert sev(rig) == "medium"
    rig.step({E: {"novelty": p_at(1e-4)}})
    assert sev(rig) == "high"


def test_high_by_axes_within_four_ticks():
    rig = Rig()
    rig.step({E: {"client": p_at(0.01)}})            # identity axis, e_day 0.01
    rig.step({E: {"client": 0.5}})
    rig.step({E: {"novelty": p_at(1e-4), "client": 0.5}})
    a = rig.alarm()
    assert a["severity"] == "high" and a["n_axes"] == 2
    # older than 4 ticks does not count
    rig = Rig()
    rig.step({E: {"client": p_at(0.01)}})
    for _ in range(4):
        rig.step({E: {"client": 0.5}})
    rig.step({E: {"novelty": p_at(1e-4), "client": 0.5}})
    assert sev(rig) == "medium"


def test_high_by_discrete_finding_and_critical_by_lib4():
    rig = Rig()
    rig.store.add_event(BehaviorEvent(system=S, entity=E, ts=rig.t, kind="beacon", score=1.0,
                                      severity=Severity.HIGH))
    rig.step({E: {"novelty": p_at(1e-6)}})
    assert sev(rig) == "high"
    # a non-discrete kind (incident) does not corroborate
    rig = Rig()
    rig.store.add_event(BehaviorEvent(system=S, entity=E, ts=rig.t, kind="incident", score=1.0,
                                      severity=Severity.HIGH))
    rig.step({E: {"novelty": p_at(1e-6)}})
    assert sev(rig) == "medium"
    # lib-4 match >= HIGH at the previous tick (one-tick lag) -> CRITICAL allowed
    rig = Rig()
    rig.store.add_match(SignatureMatch(system=S, entity=E, ts=rig.t - DT, signature_id="x",
                                       label="scan", category="recon", confidence=0.9,
                                       severity=Severity.HIGH))
    rig.step({E: {"novelty": p_at(1e-6)}})
    assert sev(rig) == "critical"


def test_critical_needs_three_axes():
    rig = Rig()
    rig.step({E: {"marg_int": p_at(1e-6), "novelty": p_at(1e-6), "client": p_at(1e-6)}})
    a = rig.alarm()
    assert a["severity"] == "critical" and a["n_axes"] == 3
    rig = Rig()
    rig.step({E: {"marg_int": p_at(1e-6), "novelty": p_at(1e-6)}})
    assert sev(rig) == "high"


def test_evidence_only_low_then_medium_at_2h():
    rig = Rig()
    levels = []
    for _ in range(10):
        rig.step({E: {"novelty": 0.01}}, trust=None)
        a = rig.alarm()
        levels.append(None if a is None
                      else (a["path"], a["severity"], a["evidence"] >= 2 * a["h"]))
    assert levels[3] == (F.PATH_EVIDENCE, "low", False)
    assert all(lv[1] == ("medium" if lv[2] else "low") for lv in levels[3:])
    assert levels[-1][1] == "medium"


# ======================================================================= paths
def test_accumulator_path_uses_its_own_p():
    rig = Rig()
    rig.step({E: {"cusum": p_at(1e-3), "marg_int": 0.5}}, acc={E: {"cusum": 1}})
    a = rig.alarm()
    assert F.PATH_ACC in a["paths"] and a["acc"] == ["cusum"]
    assert a["severity"] == "medium" and a["e_day_path"] == pytest.approx(1e-3, rel=1e-6)
    # an accumulator alarm whose own p is unremarkable is still an alarm (LOW)
    rig = Rig()
    rig.step({E: {"cusum": 0.4, "marg_int": 0.5}}, acc={E: {"cusum": 1, "mcusum": 0}})
    a = rig.alarm()
    assert a["path"] == F.PATH_ACC and a["severity"] == "low" and a["paths"] == [F.PATH_ACC]
    # acc_alarm 0 -> nothing
    rig = Rig()
    rig.step({E: {"cusum": 0.4, "marg_int": 0.5}}, acc={E: {"cusum": 0}})
    assert rig.alarm() is None


def test_single_tick_low_and_alpha_mult():
    rig = Rig()
    rig.step({E: {"jsd": p_at(0.02)}})            # accumulator family, no acc_alarm
    a = rig.alarm()
    assert a["path"] == F.PATH_SINGLE and a["severity"] == "low"
    rig = Rig()
    put_model(rig.store, "__org__", "__org__", "model.feedback", {"alpha_mult": {S: 0.5}})
    rig.step({E: {"jsd": p_at(0.02)}})
    assert rig.alarm() is None                    # threshold 0.03 * 0.5 = 0.015


def test_feedback_family_weights_reach_q_all():
    rig = Rig()
    rig.step({E: {"marg_int": 1e-4, "novelty": 0.5}})
    q1 = rig.v(F.Q_ALL)
    rig2 = Rig()
    put_model(rig2.store, "__org__", "__org__", "model.feedback",
              {"family_w": {"intensity": 0.1}})
    rig2.step({E: {"marg_int": 1e-4, "novelty": 0.5}})
    q2 = rig2.v(F.Q_ALL)
    assert q1 == pytest.approx(combine.whmp([1e-4, 0.5]), rel=1e-4)
    assert q2 == pytest.approx(combine.whmp([1e-4, 0.5], [0.1, 1.0]), rel=1e-4)


def test_calib_health_weight_mult_is_used():
    rig = Rig()
    rig.store.add_derived(DerivedMetric(name="behavior.calib_health",
                                        value={"t2": {"weight_mult": 0.25}}, ts=rig.t,
                                        system=S, entity="__system__", window_s=900,
                                        kind=MetricKind.CATEGORICAL))
    rig.step({E: {"marg_int": 0.5, "t2": 1e-3}})
    assert rig.pf()["intensity"] == pytest.approx(combine.whmp([0.5, 1e-3], [1.0, 0.25]))


# ================================================================ reading rules
def test_self_peer_down_and_up():
    rig = Rig()
    rig.step({E: {"novelty": p_at(1e-3), "peer": 0.5}})
    a = rig.alarm()
    assert a["severity"] == "low" and "self_peer_down" in a["rules"]
    rig = Rig()
    rig.step({E: {"novelty": p_at(2e-3), "peer": p_at(2e-3)}})
    a = rig.alarm()
    assert a["severity"] == "high" and "self_peer_up" in a["rules"]
    # peer p <= 0.1 and not significant: no change
    rig = Rig()
    rig.step({E: {"novelty": p_at(1e-3), "peer": 0.05}})
    assert sev(rig) == "medium"


def _flag(rig: Rig, groups: Dict[str, int], e: str = E) -> None:
    rig.store.add_derived(DerivedMetric(name="behavior.common.flag", value=groups, ts=rig.t,
                                        system=S, entity=e, window_s=900,
                                        kind=MetricKind.CATEGORICAL))


def test_common_mode_lowers_volume_only_one_level():
    rig = Rig()
    _flag(rig, {"volume": 1, "app": 0})
    rig.step({E: {"marg_int": p_at(1e-3)}})
    a = rig.alarm()
    assert a["severity"] == "low" and "common_mode" in a["rules"]
    # one level only: volume-only e_day 1e-6 is MEDIUM (cap), lowered once to LOW
    rig = Rig()
    _flag(rig, {"volume": 1})
    rig.step({E: {"marg_int": p_at(1e-6), "t2": p_at(1e-6), "budget_vol": p_at(1e-6)}},
             acc={E: {"budget_vol": 1}})
    assert sev(rig) == "low"
    # app-error axis maps from the 'app' group; transport too
    rig = Rig()
    _flag(rig, {"app": 1, "transport": 1})
    rig.step({E: {"marg_shape": p_at(1e-3)}}, axes={E: {"marg_shape": ["app", "transport"]}})
    a = rig.alarm()
    assert a["axes"] == ["app_error", "transport"] and a["severity"] == "low"
    # never categorical
    rig = Rig()
    _flag(rig, {"volume": 1, "categorical": 1})
    rig.step({E: {"novelty": p_at(1e-3)}})
    assert sev(rig) == "medium"
    # unflagged volume is untouched
    rig = Rig()
    _flag(rig, {"volume": 0})
    rig.step({E: {"marg_int": p_at(1e-3)}})
    assert sev(rig) == "medium"


def test_schedule_shift_caps_temporal_at_low():
    rig = Rig()
    rig.step({E: {"offhours": p_at(1e-3)}}, acc={E: {"offhours": 1}})
    assert sev(rig) == "medium"
    rig = Rig()
    rig.store.add_event(BehaviorEvent(system=S, entity=E, ts=rig.t - 3600, kind="schedule_shift",
                                      score=0.5))
    rig.step({E: {"offhours": p_at(1e-3)}}, acc={E: {"offhours": 1}})
    a = rig.alarm()
    assert a["severity"] == "low" and "schedule_shift" in a["rules"]


def test_detector_axes_refine_defaults():
    rig = Rig()
    rig.step({E: {"cusum": p_at(1e-3)}}, axes={E: {"cusum": ["app", "breadth"]}},
             acc={E: {"cusum": 1}})
    assert rig.alarm()["axes"] == ["app_error", "breadth"]


def test_class_keys_are_scored_with_class_rules():
    rig = Rig(entities=(E,))
    ck = "class:r1"
    rig.store.register_entity(S, ck)
    rig.step({ck: {"class_int": p_at(1e-5)}})
    a = rig.alarm(ck)
    assert a["severity"] == "low" and "class_intensity_only" in a["rules"]
    assert not math.isnan(rig.v(F.Q_ALL, ck))
    rig.step({ck: {"class_shape": p_at(1e-5)}}, axes={ck: {"class_shape": ["transport"]}})
    a = rig.alarm(ck)
    assert a["severity"] == "low" and "class_system_wide" in a["rules"]
    ck2 = "class:r2"                                 # fresh key: no axes history
    rig.store.register_entity(S, ck2)
    rig.step({ck2: {"class_novel": p_at(1e-5)}}, acc={ck2: {"class_novel": 1}})
    assert sev(rig, ck2) == "medium"


# =========================================================================== BH
def test_bh_threshold_function():
    assert F.bh_threshold([]) == -1.0
    assert F.bh_threshold([0.5, NAN, 0.9]) == -1.0
    assert F.bh_threshold([0.01, 0.02, 0.5], 0.05) == pytest.approx(0.02)   # 0.02 <= 2*0.05/3
    assert F.bh_threshold([4e-4] + [0.9] * 99, 0.05) == pytest.approx(4e-4)
    assert F.bh_threshold([6e-4] + [0.9] * 99, 0.05) == -1.0


def test_bh_removes_low_single_tick_across_many_entities():
    dt = 3600.0
    ents = [f"10.0.{i // 250}.{i % 250}" for i in range(500)]
    rig = Rig(entities=ents, dt=dt)
    pv = {e: {"marg_int": 0.5} for e in ents}
    pv[ents[0]] = {"jsd": p_at(0.02, dt)}              # LOW single tick, no q_inst
    pv[ents[1]] = {"jsd": p_at(2.9e-3, dt)}            # MEDIUM, fails BH too
    rig.step(pv)
    assert F.bh_threshold(rig.v(F.Q_ALL, e) for e in ents) == -1.0
    assert rig.alarm(ents[0]) is None
    assert sev(rig, ents[1]) == "medium"               # BH is for the LOW path only
    # alone in its system the same LOW survives
    rig = Rig(dt=dt)
    rig.step({E: {"jsd": p_at(0.02, dt)}})
    assert sev(rig) == "low"


# ================================================================ edge cases
def test_empty_store():
    store = make_store()
    assert run_engine(FusionEngine(), store, T0) == 0
    store.register_entity(S, E)
    assert run_engine(FusionEngine(), store, T0) == 0
    assert store.get_model(S, E, m_calib.MODEL) is None


def test_silent_entity_after_scoring_writes_nan():
    rig = Rig()
    for _ in range(4):
        rig.step({E: {"novelty": 0.01}}, trust=None)
    S_before = rig.v(F.EVIDENCE)
    assert S_before > 0
    rig.step({})                                       # nothing scored this tick
    assert math.isnan(rig.v(F.Q_ALL)) and math.isnan(rig.v(F.Q_INST))
    assert math.isnan(rig.v(F.E_DAY))
    assert rig.v(F.EVIDENCE) == pytest.approx(S_before, rel=1e-6)   # carried, not reset
    assert rig.alarm() is None and rig.pf() is None


def test_all_nan_row_is_unscored():
    rig = Rig()
    rig.step({E: {"marg_int": NAN, "novelty": NAN}})
    assert math.isnan(rig.v(F.Q_ALL)) and math.isnan(rig.v(F.E_DAY))
    assert rig.alarm() is None and rig.pf() is None
    assert rig.v(F.EVIDENCE) == 0.0


def test_degraded_family_and_pipeline_degraded_event():
    rig = Rig()
    deg = {E: {"marg_int": "stale", "t2": "stale", "novelty": "stale"}}
    rig.step({E: {"marg_int": NAN, "t2": NAN, "novelty": NAN, "seq": 0.4}}, degraded=deg)
    pf = rig.pf()
    assert math.isnan(pf["intensity"]) and math.isnan(pf["categorical"])
    assert pf["sequence"] == pytest.approx(0.4, rel=1e-6)
    evs = rig.store.events(S, "__system__", kinds=("pipeline_degraded",))
    assert len(evs) == 1 and evs[0].extra["entity"] == E
    assert set(evs[0].extra["families"]) == {"intensity", "categorical"}
    rig.step({E: {"marg_int": NAN, "t2": NAN, "novelty": NAN, "seq": 0.4}}, degraded=deg)
    assert len(rig.store.events(S, "__system__", kinds=("pipeline_degraded",))) == 1


def test_calibration_failure_writes_nan():
    rig = Rig()
    rig.step({E: {"novelty": 0.3}})
    rig.store.put_health(F.B24_ENGINE, {"last_error_ts": rig.t})
    rig.step({E: {"novelty": p_at(1e-6)}})
    assert math.isnan(rig.v(F.Q_ALL)) and rig.alarm() is None


def test_training_emits_no_alarm_but_learns():
    rig = Rig()
    for _ in range(8):
        rig.step({E: {"marg_int": p_at(1e-6), "novelty": p_at(1e-6)}}, trust=None,
                 training=True)
        assert rig.alarm() is None and rig.n == 0
        assert rig.v(F.Q_ALL) == rig.v(F.Q_ALL)           # computed
    assert rig.store.events() == []
    meta = rig.store.get_model(S, E, m_calib.MODEL)["meta"]
    # missing trust counts as 1 in training: rows <= now - 4 ticks were admitted
    assert meta["state"]["n_admit"] == 4
    assert sum(len(r) for r in F.meta_rings(meta).values()) == 8   # inst + all


def test_q_recomputable_from_snapshot():
    """B29 contract: q is m_calib.p_from_snapshot on the meta ring with pm = raw p."""
    rig = Rig()
    rng = np.random.default_rng(0)
    for _ in range(80):
        rig.step({E: {"marg_int": float(rng.random()), "novelty": float(rng.random())}})
    ts = rig.t - rig.dt
    model = rig.store.get_model(S, E, m_calib.MODEL)
    row = emit.read_array(rig.store, S, E, emit.P, ts)
    fz = F.fuse(row)
    st = F.calib.stratum_key(make_tctx(ts)["daypart"], 900)
    for kind, p_raw, name in ((F.META_ALL, fz.p_all, F.Q_ALL), (F.META_INST, fz.p_inst, F.Q_INST)):
        q = m_calib.p_from_snapshot(model, kind, st, F.meta_score(p_raw),
                                    m_calib.uniform(S, E, kind, ts), pm=p_raw)
        assert m_calib.issued(q) == pytest.approx(rig.v(name), rel=1e-6)
    assert m_calib.ring_size(model, F.META_ALL, st) > 0


def test_perf_per_entity_tick():
    ents = [f"10.0.2.{i}" for i in range(20)]
    rig = Rig(entities=ents)
    rng = np.random.default_rng(1)
    dets = ["marg_int", "marg_shape", "peer", "t2", "novelty", "seq", "cusum", "offhours"]
    for _ in range(5):
        rig.step({e: {d: float(rng.random()) for d in dets} for e in ents})
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        rig.step({e: {d: float(rng.random()) for d in dets} for e in ents})
    per = (time.perf_counter() - t0) / (n * len(ents))
    assert per < 5e-3, f"{per * 1e3:.2f} ms per entity-tick"
