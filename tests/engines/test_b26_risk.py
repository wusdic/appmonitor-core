"""B26 RiskEngine: docs/lib3/engines.md B26 unit tests (a)-(d) plus edges.

The engine is driven in isolation: each test plays B25 (behavior.p_family,
behavior.alarm), the discrete producers (store.events), lib-4
(store.matches) and B27 (store.incidents) for a tick, then runs B26 through
tests/helpers.run_engine (strict). The 60-day nulls of (b) drive the pure
RiskState core the engine uses, so they stay fast at 86 400 ticks.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Optional

import numpy as np
import pytest

from helpers import DT, T0, make_store, put_model, run_engine

from app.engines.behavior import risk as R
from app.engines.behavior.risk import RiskEngine, RiskState
from app.models.schema import (BehaviorEvent, DerivedMetric, Incident, MetricKind, Severity,
                               SignatureMatch)

S, E = "sys", "10.0.0.1"
NAN = float("nan")
NULL_FAMILIES = ["intensity", "shape", "peer", "temporal", "categorical", "breadth",
                 "sequence", "change"]


def p_at(e_day: float, dt: float = DT) -> float:
    return e_day * dt / 86400.0


def put_pf(store, e: str, ts: float, pf: Dict[str, float], dt: float = DT) -> None:
    store.add_derived(DerivedMetric(name="behavior.p_family", value=dict(pf), ts=ts, system=S,
                                    entity=e, window_s=int(dt), kind=MetricKind.CATEGORICAL))


def put_dict(store, e: str, name: str, ts: float, v: Dict, dt: float = DT) -> None:
    store.add_derived(DerivedMetric(name=name, value=dict(v), ts=ts, system=S, entity=e,
                                    window_s=int(dt), kind=MetricKind.CATEGORICAL))


def ev(ts: float, kind: str, e: str = E, **kw) -> BehaviorEvent:
    return BehaviorEvent(system=S, entity=e, ts=ts, kind=kind, score=1.0, **kw)


def risk_at(store, ts: float, e: str = E) -> float:
    row = store.vec_at(S, e, "behavior.risk", ts)
    return float(row[0]) if row is not None else NAN


def extra(store, e: str = E) -> Dict:
    return store.profile(S, e).extra["risk"]


class Rig:
    def __init__(self, entities=(E,), dt: float = DT, config: Optional[Dict] = None) -> None:
        self.store = make_store()
        self.eng = RiskEngine()
        self.dt = dt
        self.t = T0
        self.config = config
        for e in entities:
            self.store.register_entity(S, e)

    def tick(self, pf: Optional[Dict[str, Dict[str, float]]] = None, dt: Optional[float] = None,
             training: bool = False, before=None) -> float:
        """Advance to the next tick (t += dt, first tick at T0) and run B26."""
        dt = self.dt if dt is None else dt
        if getattr(self, "_started", False):
            self.t += dt
        self._started = True
        for e, v in (pf or {}).items():
            put_pf(self.store, e, self.t, v, dt)
        if before is not None:
            before(self.store, self.t)
        run_engine(self.eng, self.store, self.t, training=training, dt=dt, config=self.config)
        return self.t

    def L(self, e: str = E) -> float:
        return self.eng._states[self.store][(S, e)].total()

    def state(self, e: str = E) -> RiskState:
        return self.eng._states[self.store][(S, e)]


# ------------------------------------------------------------------ spec (a)
def test_a_three_families_medium_then_second_stage_high():
    rig = Rig()
    p = p_at(0.01)
    pf = {"shape": p, "categorical": p, "change": p}       # W = 1.5 each, stage behavior
    for _ in range(11):
        rig.tick({E: pf})
    base = Rig()
    for _ in range(12):
        base.tick({E: pf})
    L12 = base.L()
    assert 44.0 <= L12 <= 52.0                     # 9 x 5.5 = 49.5 before 3 h of decay
    r12 = risk_at(base.store, base.t)
    assert r12 >= 30.0 and extra(base.store)["tier"] == "medium"
    assert extra(base.store)["stages"] == ["behavior"]

    def first_seen(store, ts):
        store.add_event(ev(ts, "first_seen", extra={"tier": "class", "dim": "path",
                                                    "value": "/hr/export"}))
    rig.tick({E: pf}, before=first_seen)
    r = risk_at(rig.store, rig.t)
    assert r >= 60.0 and abs(r - 71.0) < 4.0
    x = extra(rig.store)
    assert x["tier"] == "high" and x["M"] == pytest.approx(1.3)
    assert set(x["stages"]) == {"behavior", "c2"}   # class-tier novelty = low prevalence
    assert any(t["key"] == "event:first_seen@class" for t in x["top_reasons"])


# ------------------------------------------------------------------ spec (b)
@pytest.mark.parametrize("dt", [900.0, 60.0])
def test_b_null_60_days_risk_below_medium(dt):
    rng = np.random.default_rng(7 if dt == 900 else 11)
    n = int(60 * 86400 / dt)
    P = rng.random((n, len(NULL_FAMILIES)))
    st = RiskState()
    Ls = np.empty(n)
    rs = np.empty(n)
    for i in range(n):
        now = T0 + i * dt
        st.decay_to(now)
        st.add_families(now, dt, dict(zip(NULL_FAMILIES, P[i])))
        st.prune(now)
        Ls[i] = st.total()
        rs[i] = st.score(now, {}, 1.0)
    burn = int(3 * 86400 / dt)                     # 72-h components need ~3 d to settle
    assert np.mean(rs[burn:] < 30.0) >= 0.9999
    assert 5.0 <= Ls[burn:].mean() <= 10.0
    assert np.mean(rs[burn:]) < 15.0              # ~11 at the mean L


def test_b_engine_null_matches_core_and_stays_low():
    rng = np.random.default_rng(3)
    rig = Rig(entities=("10.0.0.1", "10.0.0.2"))
    core = RiskState()
    for i in range(3 * 96):
        pf = dict(zip(NULL_FAMILIES, rng.random(len(NULL_FAMILIES))))
        t = rig.tick({"10.0.0.1": pf, "10.0.0.2": pf})
        core.decay_to(t)
        core.add_families(t, DT, pf)
        core.prune(t)
        assert risk_at(rig.store, t) < 30.0
        assert risk_at(rig.store, t) == pytest.approx(core.score(t, {}, 1.0), rel=1e-5, abs=1e-4)


# ------------------------------------------------------------------ spec (c)
def test_c_36_repeats_of_one_key_at_most_twice_a_single():
    once = Rig()
    once.tick(before=lambda s, ts: s.add_event(ev(ts, "rare_access", dedupe_key="path=/salary")))
    single = once.L()
    assert single == pytest.approx(15.0)
    rig = Rig()
    for i in range(36):
        rig.tick(before=lambda s, ts: s.add_event(ev(ts, "rare_access",
                                                     dedupe_key="path=/salary")))
    comp = rig.state().L["event:rare_access"]
    assert comp <= 2.0 * single
    # distinct keys are not damped against each other
    other = Rig()
    other.tick(before=lambda s, ts: [s.add_event(ev(ts, "rare_access", dedupe_key=f"k{j}"))
                                     for j in range(3)])
    assert other.L() == pytest.approx(45.0)


def test_c_lib4_matches_damped_and_lagged():
    rig = Rig()

    def match(store, ts):
        store.add_match(SignatureMatch(system=S, entity=E, ts=ts, signature_id="sig.exfil",
                                       label="bulk upload", category="exfil", confidence=0.8,
                                       severity=Severity.HIGH))
    t0 = rig.tick(before=match)
    assert rig.L() == 0.0                          # lib-4 runs after B26: one-tick lag
    rig.tick()
    assert rig.L() == pytest.approx(24.0, rel=1e-3)          # high 30 x 0.8
    assert "exfiltration" in extra(rig.store)["stages"]
    assert rig.state().H["lib4:sig.exfil"] == R.HALF_LIFE_S["exfil"]
    for _ in range(40):                            # the same signature every tick
        rig.tick(before=match)
    assert rig.state().L["lib4:sig.exfil"] <= 2 * 24.0
    # an info match weighs nothing
    info = Rig()
    info.tick(before=lambda s, ts: s.add_match(SignatureMatch(
        system=S, entity=E, ts=ts, signature_id="sig.browse", label="browse",
        category="browse", confidence=1.0, severity=Severity.INFO)))
    info.tick()
    assert info.L() == 0.0
    assert t0 == T0


def test_habitual_routine_lib4_activity_stops_counting():
    """Integration: lib-4 grades activities, so a routine 'low' match (a
    login, a form write) of an entity that has done it for a day is habitual
    and weighs nothing; a new routine activity counts for its first day and
    high / critical matches always count. Habits survive the warm-up reset."""
    rig = Rig()

    def routine(sig, sev=Severity.LOW, cat="auth"):
        return lambda store, ts: store.add_match(SignatureMatch(
            system=S, entity=E, ts=ts, signature_id=sig, label=sig, category=cat,
            confidence=1.0, severity=sev))
    for _ in range(100):                                  # > 24 h of routine logins, warm-up
        rig.tick(before=routine("auth_login"), training=True)
    rig.tick(before=routine("auth_login"))                # first live tick: evidence reset
    rig.tick(before=routine("auth_login"))
    assert rig.state().L.get("lib4:auth_login", 0.0) == 0.0
    rig.tick(before=routine("admin_operation", cat="admin"))   # a new activity counts
    rig.tick()
    assert rig.state().L["lib4:admin_operation"] == pytest.approx(5.0, rel=1e-2)
    rig.tick(before=routine("auth_bruteforce", Severity.HIGH))  # never habitual
    rig.tick()
    assert rig.state().L["lib4:auth_bruteforce"] == pytest.approx(30.0, rel=1e-2)


# ------------------------------------------------------------------ spec (d)
@pytest.mark.parametrize("dt_after", [900.0, 60.0])
def test_d_risk_halves_after_half_life(dt_after):
    rig = Rig()
    t0 = rig.tick({E: {"intensity": p_at(1e-2)}})     # b = 2, H = 12 h
    L0, r0 = rig.L(), risk_at(rig.store, t0)
    assert L0 == pytest.approx(2.0) and 3.0 < r0 < 4.0
    n = int(12 * 3600 / dt_after)
    for _ in range(n):                              # silent entity: no p_family at all
        t = rig.tick(dt=dt_after)
    assert t - t0 == pytest.approx(12 * 3600)
    assert rig.L() == pytest.approx(L0 / 2, rel=1e-6)
    r1 = risk_at(rig.store, t)
    assert r1 / r0 == pytest.approx(0.5, abs=0.02)
    # 72-h novelty evidence halves only after 72 h
    nov = Rig()
    nov.tick({E: {"categorical": p_at(1e-2)}})
    nov.tick(dt=72 * 3600.0)
    assert nov.L() == pytest.approx(1.5, rel=1e-6)


# ------------------------------------------------------------------ mechanics
def test_episode_saturation_bounded_and_reset():
    rig = Rig()
    p = p_at(1e-3)                                  # b = 3 for intensity
    for _ in range(80):
        rig.tick({E: {"intensity": p}})
    assert rig.L() <= 6.3 * 3.0
    assert rig.state().streak["intensity"] == 80
    rig.tick({E: {"intensity": 0.5}})               # valid, unremarkable: episode ends
    assert "intensity" not in rig.state().streak
    before = rig.L()
    rig.tick({E: {"intensity": p}})
    assert rig.L() - before * 2 ** (-DT / 43200.0) == pytest.approx(3.0, rel=1e-5)


def test_open_incident_keeps_episode_and_suppressed_counts_quarter():
    rig = Rig()
    rig.store.put_incident(Incident(system=S, entity=E, status="open", opened=T0, last_seen=T0))
    p = p_at(1e-3)
    rig.tick({E: {"intensity": p}})
    rig.tick({E: {"intensity": 0.5}})               # the incident is the episode
    assert rig.state().streak["intensity"] == 1
    sup = Rig()
    sup.store.put_incident(Incident(system=S, entity=E, status="suppressed", opened=T0,
                                    last_seen=T0))
    sup.tick({E: {"intensity": p}}, before=lambda s, ts: s.add_event(
        ev(ts, "beacon", dedupe_key="c2.example")))
    assert sup.L() == pytest.approx(0.25 * (3.0 + 20.0))
    # a suppressed event alone also counts at 0.25
    one = Rig()
    one.tick(before=lambda s, ts: s.add_event(ev(ts, "beacon", status="suppressed")))
    assert one.L() == pytest.approx(5.0)


def test_nan_and_degraded_families_are_not_evidence():
    rig = Rig()
    p = p_at(1e-3)
    rig.tick({E: {"intensity": p}})
    rig.tick({E: {"intensity": NAN, "shape": NAN}})  # unscored: no evidence, streak kept
    assert rig.state().streak["intensity"] == 1
    assert rig.L() == pytest.approx(3.0 * 2 ** (-DT / 43200.0))
    t = rig.tick({E: {"shape": NAN}})
    assert math.isfinite(risk_at(rig.store, t))
    assert R.excess_surprise(NAN, DT) != R.excess_surprise(NAN, DT)
    assert R.excess_surprise(1.0, DT) == 0.0


def test_common_mode_volume_discounted():
    rig = Rig()
    rig.tick({E: {"intensity": p_at(1e-3), "shape": p_at(1e-3)}},
             before=lambda s, ts: put_dict(s, E, "behavior.common.flag", ts, {"volume": 1}))
    assert rig.state().L["family:intensity"] == pytest.approx(0.3 * 3.0)
    assert rig.state().L["family:shape"] == pytest.approx(4.5)


def test_refined_axes_give_stages_and_critical_alarm_slows_decay():
    rig = Rig()

    def ctx_(store, ts):
        put_dict(store, E, "behavior.axes", ts, {"novelty": ["exfil"], "marg_int": ["volume"]})
        put_dict(store, E, "behavior.alarm", ts, {"path": "single_tick",
                                                  "severity": "critical"})
    rig.tick({E: {"categorical": p_at(1e-6), "intensity": p_at(1e-6)}}, before=ctx_)
    x = extra(rig.store)
    assert set(x["stages"]) == {"exfiltration", "behavior"}
    assert x["M"] == pytest.approx(1.3)
    assert rig.state().H["family:intensity"] == R.HALF_LIFE_S["critical"]
    assert rig.state().H["family:categorical"] == R.HALF_LIFE_S["novelty"]  # slower kept
    # weak evidence (e_day 0.5) adds L but earns no stage
    weak = Rig()
    weak.tick({E: {"breadth": p_at(0.5)}})
    assert weak.L() > 0 and extra(weak.store)["stages"] == []


def test_discrete_weights_tiers_and_adoption():
    rig = Rig()

    def evs(store, ts):
        store.add_event(ev(ts, "first_seen", extra={"tier": "system", "token": "sni=a.cn"}))
        store.add_event(ev(ts, "first_seen", extra={"tier": "entity", "token": "sni=b.cn"}))
        store.add_event(ev(ts, "first_seen", extra={"tier": "class", "token": "sni=c.cn",
                                                    "adopted": True}))
        store.add_event(ev(ts, "client_impersonation", severity=Severity.HIGH))
        store.add_event(ev(ts, "class_adopted"))      # INFO kind: no weight
    rig.tick(before=evs)
    assert rig.L() == pytest.approx(15 + 3 + 0.8 + 20)
    assert "identity" in extra(rig.store)["stages"]
    # events are consumed once (next tick does not re-count them)
    rig.tick()
    assert rig.L() == pytest.approx((15 + 3 + 0.8 + 20) * 2 ** (-DT / (72 * 3600.0)))


# ------------------------------------------------------------------ classes, system
def test_class_and_system_risk():
    ents = [f"10.0.0.{i}" for i in range(1, 6)]
    rig = Rig(entities=ents)
    put_model(rig.store, "__org__", "__org__", "model.class",
              {"assign": {f"{S}|{e}": {"role": "r1", "prob": 1.0} for e in ents}})
    pf = {e: {"intensity": p_at(10 ** -(i + 1))} for i, e in enumerate(ents)}
    t = rig.tick(pf)
    rs = sorted((risk_at(rig.store, t, e) for e in ents), reverse=True)
    cls = risk_at(rig.store, t, "class:r1")
    assert cls == pytest.approx(np.mean(rs[:3]), rel=1e-5)
    assert extra(rig.store, "class:r1")["members"] == pytest.approx(np.mean(rs[:3]), abs=1e-3)
    assert risk_at(rig.store, t, "__system__") == pytest.approx(max(rs + [cls]), rel=1e-6)
    # the class key's own detectors dominate when stronger
    t = rig.tick({"class:r1": {"shape": p_at(1e-8)}})
    own = extra(rig.store, "class:r1")["own"]
    assert own > np.mean(sorted((risk_at(rig.store, t, e) for e in ents), reverse=True)[:3])
    assert risk_at(rig.store, t, "class:r1") == pytest.approx(own, abs=1e-3)
    assert risk_at(rig.store, t, "__system__") >= risk_at(rig.store, t, "class:r1") - 1e-4


def test_tier_hysteresis():
    assert R.tier_of(29.9) == "low" and R.tier_of(30) == "medium"
    assert R.tier_of(85) == "critical"
    assert R.tier_of(55, "high") == "high"          # kept down to 50
    assert R.tier_of(49.9, "high") == "medium"
    assert R.tier_of(15, "high") == "low"
    assert R.tier_of(21, "medium") == "medium"
    assert R.tier_of(76, "critical") == "critical" and R.tier_of(74, "critical") == "high"
    # engine: rises into high, decays to ~55 and stays high (one stage, 72-h decay)
    rig = Rig()
    for _ in range(3):
        rig.tick({E: {"identity": p_at(1e-12)}})
    assert extra(rig.store)["tier"] == "high"
    while risk_at(rig.store, rig.t) >= 55.0:
        rig.tick()
    r = risk_at(rig.store, rig.t)
    assert 50.0 <= r < 60.0 and extra(rig.store)["tier"] == "high"


def test_criticality_and_feedback_multiplier():
    cfg = {"ip_classes": [{"name": "dc", "cidrs": ["10.0.0.0/24"], "systems": [S],
                           "criticality": 2.0}]}
    base, crit = Rig(), Rig(config=cfg)
    for rig in (base, crit):
        rig.tick({E: {"shape": p_at(1e-3)}})
    rb, rc = risk_at(base.store, base.t), risk_at(crit.store, crit.t)
    x = 4.5 / 60.0
    assert rb == pytest.approx(100 * (1 - math.exp(-x)), rel=1e-5)
    assert rc == pytest.approx(100 * (1 - math.exp(-2 * x)), rel=1e-5)
    over = Rig(config={"criticality": {f"{S}|{E}": 9.0}})      # clipped to 2
    over.tick({E: {"shape": p_at(1e-3)}})
    assert risk_at(over.store, over.t) == pytest.approx(rc, rel=1e-6)
    fb = Rig()
    put_model(fb.store, "__org__", "__org__", "model.feedback",
              {"detector_prec": {"shape|*": [0, 30]}})           # all fp -> pi = 0.2 floor
    fb.tick({E: {"shape": p_at(1e-3)}})
    assert risk_at(fb.store, fb.t) == pytest.approx(100 * (1 - math.exp(-0.2 * x)), rel=1e-5)


# ------------------------------------------------------------------ edges
def test_empty_store_and_silent_entity():
    store = make_store()
    eng = RiskEngine()
    assert run_engine(eng, store, T0) == 0
    assert store.systems() == []
    rig = Rig()
    t = rig.tick()
    assert risk_at(rig.store, t) == 0.0
    x = extra(rig.store)
    assert x["tier"] == "low" and x["top_reasons"] == [] and x["stages"] == []
    assert risk_at(rig.store, t, "__system__") == 0.0


def test_training_emits_no_events_but_accumulates():
    rig = Rig()
    t = rig.tick({E: {"shape": p_at(1e-4)}}, training=True)
    assert rig.store.events() == []
    assert risk_at(rig.store, t) > 0.0


def test_cadence_switch_900_to_60_is_wall_clock():
    a, b = Rig(), Rig()
    for rig in (a, b):
        rig.tick({E: {"change": p_at(1e-3)}})
    a.tick(dt=900.0)
    for _ in range(15):
        b.tick(dt=60.0)
    assert a.t == b.t and a.L() == pytest.approx(b.L(), rel=1e-9)
    # the same e_day gives the same evidence per tick at either cadence
    assert R.excess_surprise(p_at(1e-3, 60.0), 60.0) == pytest.approx(
        R.excess_surprise(p_at(1e-3, 900.0), 900.0))
    b.tick({E: {"change": p_at(1e-3, 60.0)}}, dt=60.0)
    a.tick({E: {"change": p_at(1e-3, 60.0)}}, dt=60.0)        # a's first 60-s tick
    assert a.L() == pytest.approx(b.L(), rel=1e-6)
    assert risk_at(a.store, a.t) == pytest.approx(risk_at(b.store, b.t), rel=1e-5)


def test_fusion_failure_gives_nan_and_keeps_state():
    rig = Rig()
    rig.tick({E: {"shape": p_at(1e-3)}})
    L0 = rig.L()
    rig.store.put_health("behavior.fusion", {"last_error_ts": T0 + DT, "ok": False})
    t = rig.tick({E: {"shape": p_at(1e-3)}})
    assert math.isnan(risk_at(rig.store, t))
    assert extra(rig.store)["degraded"] is True
    assert rig.L() == L0
    t = rig.tick()
    assert math.isfinite(risk_at(rig.store, t))


def test_rerun_same_tick_is_idempotent_and_clock_back_resets():
    rig = Rig()
    t = rig.tick({E: {"shape": p_at(1e-3)}})
    L0 = rig.L()
    run_engine(rig.eng, rig.store, t, dt=DT)
    assert rig.L() == L0
    # a clock that goes backwards (new run / replay) starts the key afresh
    states = rig.eng._states[rig.store]
    st = RiskEngine._state(states, S, E, t - 10 * DT)
    assert st.total() == 0.0 and st.last_ts is None
    # a new store never sees another store's state
    other = make_store()
    other.register_entity(S, E)
    run_engine(rig.eng, other, t, dt=DT)
    assert rig.eng._states[other][(S, E)].total() == 0.0


def test_perf_40_entities():
    ents = [f"10.0.1.{i}" for i in range(40)]
    rig = Rig(entities=ents)
    rng = np.random.default_rng(5)
    put_model(rig.store, "__org__", "__org__", "model.class",
              {"assign": {f"{S}|{e}": {"role": str(i % 4), "prob": 1.0}
                          for i, e in enumerate(ents)}})
    ticks = 40
    t0 = time.perf_counter()
    for _ in range(ticks):
        pf = {e: dict(zip(NULL_FAMILIES, rng.random(8))) for e in ents}
        rig.tick(pf)
    per_tick = (time.perf_counter() - t0) / ticks
    # generous: engine + test scaffolding, 40 entities + 4 classes per tick
    assert per_tick < 0.05
