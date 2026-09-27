"""B28 GovernorEngine: docs/lib3/engines.md B28 unit tests (a)-(g).

The engine is driven in isolation: each test plays B25 (behavior.alarm,
q_inst / q_all), B24 (behavior.p), the accumulator owners (behavior.acc_alarm),
B14 (model.cp episode onset, baseline_creep), B04 / B01 (feature.vec: the
level whose stationarity and dispersion the governor measures), lib-4
(store.matches), B27 (store.incidents) and B02 (model.class) for a tick, then
runs B28 through tests/helpers.run_engine (strict). Test (d) runs a real
lib/gating GatedLearner before B28 on every tick, as B03 would in the
pipeline, and checks the replayed statistics after the rollback.

Known corrections applied (helper reviewers): trust / trust_prov / quarantine
are 1-element vec rings; quarantine is 1 at the tick rollback_to is written
and 0 from the tick release is written; lib-4 matches are read with a
one-tick lag. Test (d) compares with a fit on rows <= tau-hat - dt (the rows
rollback_to = tau-hat - dt keeps; the spec text says "<= tau-hat").
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pytest

from helpers import DT, T0, make_store, put_model, run_engine

from app.engines.behavior.governor import GovernorEngine
from app.engines.behavior.lib import emit, gating
from app.engines.behavior.lib import m_governor as MG
from app.engines.behavior.lib.features import FEATURE_DIM, GROUPS
from app.models.schema import (BehaviorEvent, DerivedMetric, Incident, MetricKind, Severity,
                               SignatureMatch)

S = "sys"
E = "10.0.0.1"
DAY = 86400.0
HOUR = 3600.0
NAN = float("nan")
VOL = list(GROUPS["volume"])


# ------------------------------------------------------------------ fixtures
def put1(store, e: str, name: str, ts: float, v: float, dt: float = DT) -> None:
    store.add_vec(S, e, name, ts, np.asarray([v], dtype=np.float32), window_s=int(dt))


def put_dict(store, e: str, name: str, ts: float, v: Dict, dt: float = DT) -> None:
    store.add_derived(DerivedMetric(name=name, value=dict(v), ts=ts, system=S, entity=e,
                                    window_s=int(dt), kind=MetricKind.CATEGORICAL))


def put_alarm(store, e: str, ts: float, axes: Sequence[str] = ("volume",),
              sev: str = "medium", dt: float = DT) -> None:
    put_dict(store, e, "behavior.alarm", ts, {"path": "single_tick", "severity": sev,
                                              "axes": list(axes), "e_day": 1e-3}, dt)


def put_acc(store, e: str, ts: float, alarms: Dict[str, int], dt: float = DT) -> None:
    emit.write_scores(store, S, e, ts, {}, acc_alarm=alarms, window_s=int(dt))


def put_vec(store, e: str, ts: float, level: float, rng: np.random.Generator,
            noise: float = 0.05, dt: float = DT) -> None:
    row = np.full(FEATURE_DIM, np.nan)
    row[VOL] = level + noise * rng.standard_normal(len(VOL))
    store.add_vec(S, e, "feature.vec", ts, row.astype(np.float32), window_s=int(dt))


def match(store, e: str, ts: float, sev: Severity = Severity.HIGH) -> None:
    store.add_match(SignatureMatch(system=S, entity=e, ts=ts, signature_id="sig.exfil",
                                   label="bulk upload", category="exfil", confidence=0.9,
                                   severity=sev))


def ring(store, e: str, name: str, ts: float) -> float:
    row = store.vec_at(S, e, name, ts)
    return float(row[0]) if row is not None else NAN


def regime_events(store, e: str = E) -> List[BehaviorEvent]:
    evs = store.events(system=S, entity=e, kinds=("regime",), limit=100000)
    return sorted(evs, key=lambda x: (x.ts, x.id))


def states(store, e: str = E) -> List[str]:
    return [ev.extra["state"] for ev in regime_events(store, e)]


def control(store, e: str = E) -> Dict:
    return store.get_model(S, e, "model.control") or {}


def class_model(members: Sequence[str], rid: str = "r1") -> Dict:
    return {"assign": {f"{S}|{m}": {"role": rid, "prob": 0.9, "static": [], "pool": None}
                       for m in members},
            "roles": {rid: {"name": "workers", "members": [f"{S}|{m}" for m in members]}},
            "version": 1}


class Sim:
    """One store + one GovernorEngine, ticking at dt from T0 (first tick = T0)."""

    def __init__(self, entities: Iterable[str] = (E,), dt: float = DT, seed: int = 0,
                 config: Optional[Dict] = None) -> None:
        self.store = make_store()
        self.eng = GovernorEngine()
        self.dt = dt
        self.t = T0 - dt
        self.rng = np.random.default_rng(seed)
        self.config = config
        self.entities = list(entities)
        for e in self.entities:
            self.store.register_entity(S, e)

    def normal(self, e: str, t: float, level: float = 5.0, noise: float = 0.05,
               dt: Optional[float] = None) -> None:
        """A clean tick: unremarkable fused evidence and the usual level."""
        dt = self.dt if dt is None else dt
        q = float(self.rng.uniform(0.05, 1.0))
        put1(self.store, e, "behavior.q_inst", t, q, dt)
        put1(self.store, e, "behavior.q_all", t, q, dt)
        put_vec(self.store, e, t, level, self.rng, noise, dt)

    def shifted(self, e: str, t: float, level: float, axes: Sequence[str] = ("volume",),
                noise: float = 0.05, dt: Optional[float] = None, alarm: bool = True) -> None:
        """An alarmed tick at a new level: fusion keeps alarming while the
        learners hold, and stops once the governor has rebased them."""
        dt = self.dt if dt is None else dt
        if MG.version(self.store, S, e) > 0:
            self.normal(e, t, level, noise, dt)
            return
        put1(self.store, e, "behavior.q_inst", t, 1e-7, dt)
        put1(self.store, e, "behavior.q_all", t, 1e-7, dt)
        put_vec(self.store, e, t, level, self.rng, noise, dt)
        if alarm:
            put_alarm(self.store, e, t, axes, dt=dt)

    def tick(self, before: Optional[Callable] = None, dt: Optional[float] = None,
             training: bool = False) -> float:
        dt = self.dt if dt is None else dt
        self.t += dt
        if before is not None:
            before(self.store, self.t)
        run_engine(self.eng, self.store, self.t, training=training, dt=dt, config=self.config)
        return self.t

    def run(self, n: int, before: Optional[Callable] = None, dt: Optional[float] = None,
            training: bool = False) -> List[float]:
        return [self.tick(before, dt, training) for _ in range(n)]

    def regime(self, e: str = E) -> str:
        return MG.regime(self.store, S, e)


# ================================================================ spec tests
def test_a_exfil_alarm_160_ticks_trust_zero_no_version_rejected_on_lib4():
    sim = Sim()
    sim.run(192, lambda st, t: sim.normal(E, t))
    attack: List[float] = []
    for i in range(160):
        def before(st, t, i=i):
            sim.shifted(E, t, 7.0, axes=("exfil",))
            if i == 100:
                match(st, E, t)                  # lib-4 HIGH at this tick, read next tick
        attack.append(sim.tick(before))
        if i < 101:
            assert sim.regime() in (MG.SUSPECT, MG.DRIFTING), (i, sim.regime())
    assert all(ring(sim.store, E, MG.TRUST, t) == 0.0 for t in attack)
    assert all(ring(sim.store, E, MG.QUARANTINE, t) == 1.0 for t in attack)
    assert int(control(sim.store).get("version") or 0) == 0
    assert MG.version(sim.store, S, E) == 0
    ev = regime_events(sim.store)
    rej = [x for x in ev if x.extra["state"] == "rejected"]
    assert len(rej) == 1 and rej[0].ts == attack[101]
    assert rej[0].extra["type"] == "exfil" and rej[0].severity == Severity.MEDIUM
    assert "accepted" not in states(sim.store)
    assert control(sim.store)["frozen"] is True
    assert sim.regime() == MG.REJECTED
    assert MG.is_quarantined(sim.store, S, E)


def test_b_six_of_eight_class_members_shift_together_accepted_near_onset():
    members = [f"10.0.1.{i}" for i in range(8)]
    movers = members[:6]
    ck = "class:r1"
    sim = Sim(entities=members, seed=1)
    put_model(sim.store, "__org__", "__org__", "model.class", class_model(members))

    def normal_all(st, t):
        for e in members:
            sim.normal(e, t)

    sim.run(192, normal_all)
    onset_t = sim.t + sim.dt

    def moved(st, t):
        for e in members:
            if e in movers:
                sim.shifted(e, t, 5.7)
            else:
                sim.normal(e, t)
        if MG.version(st, S, ck) == 0:
            put_alarm(st, ck, t, ("volume",), sev="low")  # B18 class_int, capped LOW

    ticks = sim.run(192, moved)                            # 2 d
    for e in movers:
        acc = [x for x in regime_events(sim.store, e) if x.extra["state"] == "accepted"]
        assert len(acc) == 1, e
        c = control(sim.store, e)
        assert c["version"] == 1
        assert abs(c["rebase_from"] - onset_t) <= 4 * DT
        # accepted after T_type (1 d) of time in regime, and within the 2 d
        assert DAY - DT <= acc[0].ts - onset_t <= 2 * DAY
        assert acc[0].extra["evidence"].get("peer") == MG.LR_PEER
        assert c["accepted_class_change"] == ck
        assert MG.regime(sim.store, S, e) == MG.NORMAL
        assert ring(sim.store, e, MG.TRUST, ticks[-1]) > 0.0
    for e in members[6:]:
        assert states(sim.store, e) == []
        assert int(control(sim.store, e).get("version") or 0) == 0
    # the class key accepts at class level after 24 h of concordance
    cc = control(sim.store, ck)
    assert cc.get("version") == 1
    acc_cls = cc["accepted_class_change"]
    assert isinstance(acc_cls, dict) and sorted(acc_cls["members"]) == sorted(members)
    cls_acc = [x for x in regime_events(sim.store, ck) if x.extra["state"] == "accepted"]
    assert len(cls_acc) == 1 and cls_acc[0].ts - onset_t >= DAY - DT


def test_c_single_entity_doubling_drifting_low_then_accepted_at_1d():
    others = ["10.0.2.2", "10.0.2.3", "10.0.2.4"]
    ents = [E] + others
    sim = Sim(entities=ents, seed=2)
    put_model(sim.store, "__org__", "__org__", "model.class", class_model(ents))

    def normal_all(st, t):
        for e in ents:
            sim.normal(e, t)

    sim.run(192, normal_all)
    onset_t = sim.t + sim.dt

    def doubled(st, t):
        sim.shifted(E, t, 5.0 + math.log(2.0))
        put_dict(st, E, "behavior.id", t, {"posterior_self": 0.97})
        for e in others:
            sim.normal(e, t)

    sim.run(120, doubled)
    ev = regime_events(sim.store)
    st = [x.extra["state"] for x in ev]
    assert st[0] == "suspect" and st[1] == "drifting"
    drift = ev[1]
    assert drift.severity == Severity.LOW and drift.extra["type"] == "intensity"
    assert drift.ts - onset_t >= HOUR
    acc = [x for x in ev if x.extra["state"] == "accepted"]
    assert len(acc) == 1
    # T_type = 1 d; a Mann-Kendall false rejection (10 % by design) can delay
    # the full time evidence by a few ticks (eval: within 1.5 d)
    assert DAY - DT <= acc[0].ts - onset_t <= 1.5 * DAY
    ex = acc[0].extra
    assert ex["type"] == "intensity" and ex["p_legit"] >= 0.9
    assert "peer" not in ex["evidence"]                   # no corroboration
    assert ex["evidence"]["dispersion"] == MG.LR_DISPERSION
    assert ex["evidence"]["id_self"] == MG.LR_ID_SELF
    c = control(sim.store)
    assert c["version"] == 1 and abs(c["rebase_from"] - (onset_t - DT)) < 1e-6
    # nothing about the others
    for e in others:
        assert states(sim.store, e) == []


def _toy_learner() -> gating.GatedLearner:
    """A stand-in for B03's current anchor: undecayed (W, sum x) of bytes_up."""
    def fetch(store, s, e, ts):
        v = store.vec_at(s, e, "feature.vec", ts)
        return None if v is None or not np.isfinite(v[VOL[0]]) else float(v[VOL[0]])

    return gating.GatedLearner(
        name="baseline.current", init=lambda: (0.0, 0.0),
        update=lambda st, x, w: (st[0] + w, st[1] + w * x), fetch=fetch,
        dump=lambda st: list(st), load=lambda b: tuple(b))


def test_d_slow_ramp_rollback_to_onset_and_baseline_equals_fit_before_onset():
    sim = Sim(seed=3)
    lr = _toy_learner()
    state, gate = lr.init(), gating.GateState()
    xs: Dict[float, float] = {}
    n_clean, n_ramp = 150, 40

    def before(st, t):
        nonlocal state, gate
        k = round((t - T0) / DT)
        level = 5.0 + (0.004 * (k - n_clean) if k >= n_clean else 0.0)   # slow, no alarm
        sim.normal(E, t, level)
        xs[t] = float(st.vec_at(S, E, "feature.vec", t)[VOL[0]])
        put1(st, E, "feature.active", t, 1.0)
        state, gate = lr.step(st, S, E, state, gate, t, DT)             # B03 runs before B28
        st.put_model(S, E, "model.baseline", {"current": {"W": state[0], "S": state[1]}})

    sim.run(n_clean + n_ramp, before)
    t_det = sim.t + DT
    tau = T0 + n_clean * DT                                             # = t_det - 40 dt
    assert abs((t_det - tau) / DT - 40) < 1e-9

    def detect(st, t):
        before(st, t)
        put_acc(st, E, t, {"creep": 1})
        put_model(st, S, E, "model.cp", {"run": {"episode": {"onset": tau, "axes": ["volume"]},
                                                 "alarm": {"creep": 1}, "dt": DT}})
        st.add_event(BehaviorEvent(system=S, entity=E, ts=t, kind="baseline_creep", score=0.5,
                                   severity=Severity.LOW, axes=["volume"],
                                   extra={"groups": {"volume": {"slope_log": 0.38, "p": 1e-3}}}))

    sim.tick(detect)
    c = control(sim.store)
    assert c["rollback_to"] == pytest.approx(tau - DT, abs=1e-6)
    assert ring(sim.store, E, MG.QUARANTINE, t_det) == 1.0              # at the rollback tick
    assert "rollback" in states(sim.store)
    rb = [x for x in regime_events(sim.store) if x.extra["state"] == "rollback"][0]
    assert rb.ts == t_det and rb.extra["onset"] == pytest.approx(tau)

    def after(st, t):
        before(st, t)
        put_acc(st, E, t, {"creep": 1})

    sim.tick(after)                                                     # learners apply it
    base = sim.store.get_model(S, E, "model.baseline")["current"]
    keep = [x for t, x in sorted(xs.items()) if t <= tau - DT]
    assert base["W"] == pytest.approx(len(keep), abs=1e-9)
    assert base["S"] == pytest.approx(sum(keep), rel=1e-12)
    assert gate.applied["rollback_to"] == pytest.approx(tau - DT)
    assert gate.held and gate.held[0].ts == pytest.approx(tau)          # the ramp rows are held
    assert all(r.ts <= tau - DT for r in gate.journal)


def test_e_training_empty_store_trust_one_and_first_live_tick_trusted():
    sim = Sim(entities=("10.9.9.9",), seed=4)
    e = "10.9.9.9"
    train = sim.run(24, training=True)                  # nothing but the entity exists
    for t in train:
        assert ring(sim.store, e, MG.TRUST, t) == 1.0
        assert ring(sim.store, e, MG.TRUST_PROV, t) == 1.0
        assert ring(sim.store, e, MG.QUARANTINE, t) == 0.0
    assert states(sim.store, e) == []
    t = sim.tick(lambda st, t: sim.normal(e, t))        # first live tick, a normal entity
    assert ring(sim.store, e, MG.TRUST, t) > 0.0
    assert sim.regime(e) == MG.NORMAL
    # and with nothing scored at all (absence is data, not suspicion)
    t = sim.tick()
    assert ring(sim.store, e, MG.TRUST, t) == 1.0


def test_e_training_alarms_do_not_leak_into_live_and_lib4_high_zeroes_trust():
    sim = Sim(seed=5)
    sim.run(8, lambda st, t: (sim.shifted(E, t, 9.0), put_acc(st, E, t, {"cusum": 1})),
            training=True)
    assert states(sim.store) == [] and sim.regime() == MG.NORMAL
    t_m = sim.tick(lambda st, t: (sim.normal(E, t), match(st, E, t)), training=True)
    t_next = sim.tick(lambda st, t: sim.normal(E, t), training=True)
    assert ring(sim.store, E, MG.TRUST, t_m) == 1.0     # the match is read one tick later
    assert ring(sim.store, E, MG.TRUST, t_next) == 0.0
    assert ring(sim.store, E, MG.TRUST_PROV, t_next) == 0.0
    t_live = sim.tick(lambda st, t: sim.normal(E, t))
    assert ring(sim.store, E, MG.TRUST, t_live) > 0.0


def test_f_incident_closes_when_returned_even_with_high_risk():
    sim = Sim(seed=6)
    sim.run(96, lambda st, t: sim.normal(E, t))
    t_open = sim.t + DT
    inc_id = sim.store.put_incident(Incident(system=S, entity=E, status="open", opened=t_open,
                                             last_seen=t_open, risk=80.0))

    def alarm(st, t):
        sim.shifted(E, t, 6.0)
        put1(st, E, "behavior.risk", t, 80.0)

    def quiet(st, t):
        sim.normal(E, t)
        put1(st, E, "behavior.risk", t, 75.0)           # risk stays high: irrelevant

    sim.run(4, alarm)
    t_last_alarm = sim.t
    qt = sim.run(12, quiet)
    inc = sim.store.get_incident(inc_id)
    assert inc.status == "closed" and inc.close_reason == "returned"
    ev = regime_events(sim.store)
    ret = [x for x in ev if x.extra["state"] == "returned"]
    assert len(ret) == 1
    t_ret = ret[0].ts
    assert t_ret - t_last_alarm >= max(8 * DT, 2 * HOUR)
    assert t_ret - t_last_alarm <= max(8 * DT, 2 * HOUR) + DT
    c = control(sim.store)
    # release = [min(tau-hat, q_floor), t]: q_floor is the commit frontier when the
    # incident started the quarantine (t_open - D dt), so no held row is stranded
    assert c["release"][0] == pytest.approx(t_open - 4 * DT)
    assert c["release"][1] == pytest.approx(t_ret)
    # quarantine 1 up to the tick before the release, 0 from the release tick on
    assert ring(sim.store, E, MG.QUARANTINE, t_ret - DT) == 1.0
    assert ring(sim.store, E, MG.QUARANTINE, t_ret) == 0.0
    assert ring(sim.store, E, MG.TRUST, qt[-1]) > 0.0
    # B27 reads the transition from behavior.regime
    pts = sim.store.derived_tail(S, E, MG.REGIME, 8)
    assert any(p.value["state"] == "returned" and p.ts == t_ret for p in pts)
    assert sim.regime() == MG.NORMAL


def _ramp(slope_log_per_day: float, seed: int, days_after: float) -> Sim:
    """3 d clean, a ramp from r0 at `slope` log-units/day detected by creep 2 d
    later (tau-hat = r0 from B14), then `days_after` days of the ramp with the
    creep accumulator latched."""
    sim = Sim(seed=seed)
    per_day = int(DAY / DT)
    n_clean, n_pre = 3 * per_day, 2 * per_day
    r0 = T0 + n_clean * DT
    sim.r0 = r0

    def level(t):
        return 5.0 + slope_log_per_day * max(0.0, t - r0) / DAY

    sim.run(n_clean + n_pre, lambda st, t: sim.normal(E, t, level(t)))
    sim.t_det = sim.t + DT

    def detect(st, t):
        sim.normal(E, t, level(t))
        if MG.version(st, S, E) > 0:            # B14 resets its statistics on a rebase
            put_model(st, S, E, "model.cp", {"run": {"episode": {}, "alarm": {}, "dt": DT}})
            return
        put_acc(st, E, t, {"creep": 1})
        put_model(st, S, E, "model.cp", {"run": {"episode": {"onset": r0, "axes": ["volume"]},
                                                 "alarm": {"creep": 1}, "dt": DT}})
        if t == sim.t_det:
            st.add_event(BehaviorEvent(
                system=S, entity=E, ts=t, kind="baseline_creep", score=0.5,
                severity=Severity.LOW, axes=["volume"],
                extra={"groups": {"volume": {"slope_log": slope_log_per_day, "p": 1e-4}}}))

    sim.run(int(days_after * per_day), detect)
    return sim


def test_g_fast_ramp_never_accepted():
    sim = _ramp(math.log(1.15), seed=7, days_after=3.0)
    st = states(sim.store)
    assert st[0] == "suspect" and "drifting" in st
    assert "accepted" not in st and "rejected" not in st
    assert sim.regime() == MG.DRIFTING
    assert int(control(sim.store).get("version") or 0) == 0
    m = MG.get(sim.store, S, E)
    assert m["type"] == "ramp" and m["p_legit"] < 0.9


def test_g_gentle_ramp_accepted_after_one_day_of_stationarity():
    sim = _ramp(math.log(1.02), seed=8, days_after=1.6)
    ev = regime_events(sim.store)
    st = [x.extra["state"] for x in ev]
    assert st[0] == "suspect"
    assert st.count("suspect") == 1                      # at most one regime episode
    acc = [x for x in ev if x.extra["state"] == "accepted"]
    assert len(acc) == 1
    assert DAY - DT <= acc[0].ts - sim.t_det <= 1.5 * DAY
    ex = acc[0].extra
    assert ex["type"] == "ramp" and ex["evidence"]["prior"] == MG.PRIOR["ramp"]
    c = control(sim.store)
    assert c["version"] == 1
    assert c["rebase_from"] == pytest.approx(sim.r0)
    assert c["allow_drift"] == pytest.approx(math.log(1.02))
    # the rollback reached back to the onset B14 reported
    assert c["rollback_to"] == pytest.approx(sim.r0 - DT)
    # never above LOW
    assert all(x.severity in (Severity.INFO, Severity.LOW) for x in ev)
