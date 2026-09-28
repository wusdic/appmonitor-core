"""Habituation of recurring HIGH lib-4 matches (lib/m_habit; B26 learns and
writes it, B27 / B28 read it). Lead decision, evaluation round 4.

Evidence it fixes: the sanctioned nightly backup hosts (pack L15) matched
`bulk_upload` (HIGH) every night; B26 counted it at full weight (risk 80-100,
CRITICAL in every run) and B28 zeroed the warm-up trust of every backup
tick, so B13 never held a trusted night hour. Each test below fails with the
habituation switched off (HIGH_HABIT_MULT = 1 / _habitual_high -> False).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from helpers import T0, make_store, run_engine

from app.engines.behavior import governor as GV
from app.engines.behavior import risk as R
from app.engines.behavior.governor import GovernorEngine
from app.engines.behavior.lib import m_governor as MG
from app.engines.behavior.lib import m_habit as HB
from app.engines.behavior.lib import timebins as TB
from app.engines.behavior.risk import RiskEngine
from app.models.schema import MetricKind, RawMetric, Severity, SignatureMatch

S, E = "sys", "10.20.9.5"
TZ = "Asia/Shanghai"
DT = 3600.0
BACKUP = "10.0.0.60"
CFG = {"tz": TZ}
# T0 = 2023-11-14 22:13:20 UTC; align the rig to a local midnight
_L0 = TB.slot_of(T0, TZ) // 96
T_START = min(t for t in (T0 + k * 900.0 for k in range(200))
              if TB.slot_of(t, TZ) // 96 > _L0 and TB.slot_of(t, TZ) % 96 == 0)


def _hour(ts: float) -> int:
    return HB.local_day_hour(ts, TZ)[1]


def put_match(store, ts: float, up: float = 6e8, peers=(BACKUP,),
              sev: Severity = Severity.HIGH, sig: str = "bulk_upload") -> None:
    store.add_raw(RawMetric(name="l4.bytes_up", value=up, ts=ts, system=S, entity=E,
                            kind=MetricKind.COUNTER))
    store.add_raw(RawMetric(name="l4.peer_set", value={p: 10 for p in peers}, ts=ts,
                            system=S, entity=E, kind=MetricKind.CATEGORICAL))
    store.add_match(SignatureMatch(
        system=S, entity=E, ts=ts, signature_id=sig, label="bulk upload", category="transfer",
        confidence=1.0, severity=sev,
        evidence={"l4.bytes_up gt 3000000": up * 900.0 / DT, "derived.upload_dominance gt 3": 9.0}))


class Rig:
    """B26 (+ optionally B28) on one entity at hourly ticks from a local midnight."""

    def __init__(self, governor: bool = False) -> None:
        self.store = make_store()
        self.store.register_entity(S, E)
        self.risk = RiskEngine()
        self.gov = GovernorEngine() if governor else None
        self.t = T_START

    def tick(self, before=None, training: bool = False) -> float:
        self.t += DT
        if before is not None:
            before(self.store, self.t)
        run_engine(self.risk, self.store, self.t, training=training, dt=DT, config=CFG)
        if self.gov is not None:
            run_engine(self.gov, self.store, self.t, training=training, dt=DT, config=CFG)
        return self.t

    def nights(self, n: int, training: bool = True, hour: int = 1, **kw) -> None:
        """n days, one backup match per night in the tick ending at hour + 1."""
        for _ in range(n * 24):
            nxt = self.t + DT
            self.tick((lambda st, t: put_match(st, t, **kw)) if _hour(nxt) == hour else None,
                      training=training)

    def L(self, sig: str = "bulk_upload") -> float:
        st = self.risk._states[self.store][(S, E)]
        return st.L.get(f"lib4:{sig}", 0.0)

    def until_hour(self, hour: int, training: bool = False) -> None:
        while _hour(self.t + DT) != hour:
            self.tick(training=training)

    def one(self, hour: int, **kw) -> float:
        """Advance to `hour`, match there, run the next tick (the one-tick
        lag); returns the lib-4 L the match added."""
        self.until_hour(hour)
        l0 = self.L()
        self.tick(lambda st, t: put_match(st, t, **kw))
        self.tick()
        return self.L() - l0 * 2.0 ** (-2 * DT / R.HALF_LIFE_S["exfil"])


def verdicts(store) -> list:
    rec = HB.get(store, S, E)["sigs"]["bulk_upload"]
    return [v for _, v in sorted(rec["v"].items(), key=lambda kv: float(kv[0]))]


# ------------------------------------------------------------------ pure rule
def test_judge_learning_then_in_and_each_envelope_exit():
    rec = {"days": {}, "peers": []}
    for d in range(4):
        HB.learn(rec, d, 1, 1.5e8, 6e8, [BACKUP])
    assert HB.judge(rec, 4, 1, 1.5e8, 6e8, [BACKUP])[0] == HB.LEARNING   # 4 days < 5
    HB.learn(rec, 4, 1, 1.5e8, 6e8, [BACKUP])
    assert HB.judge(rec, 5, 1, 1.5e8, 6e8, [BACKUP]) == (HB.IN, [])
    assert HB.judge(rec, 5, 2, 1.5e8, 6e8, [BACKUP])[0] == HB.IN          # +-1 h
    for kw, word in ((dict(hour=14), "hour"), (dict(peak=5e8), "peak"),
                     (dict(amount=2e9), "daily amount"), (dict(peers=[BACKUP, "6.6.6.6"]),
                                                           "new peer")):
        a = dict(hour=1, peak=1.5e8, amount=6e8, peers=[BACKUP])
        a.update(kw)
        v, why = HB.judge(rec, 5, a["hour"], a["peak"], a["amount"], a["peers"])
        assert v == HB.OUT and word in why[0]
    # a day's running amount counts every match of the day, trusted or not
    rec["today"] = [5, 5e8]
    assert HB.judge(rec, 5, 1, 1.5e8, 6e8, [BACKUP])[0] == HB.OUT
    # no volume / peer data is not an exceedance
    assert HB.judge(rec, 6, 1, None, None, None)[0] == HB.IN


def test_slack_follows_the_trusted_spread():
    assert HB.slack([1.0]) == pytest.approx(1.5)
    assert HB.slack([1.0, 1.01, 0.99, 1.0]) == pytest.approx(1.5)        # tight: floor
    wide = [1.0, 3.0, 0.5, 2.0, 1.0]
    sd = float(np.std(np.log(wide), ddof=1))
    assert HB.slack(wide) == pytest.approx(math.exp(2 * sd))


def test_one_off_hours_are_not_a_rhythm():
    days = {"0": {"h": [1, 13]}, "1": {"h": [1]}, "2": {"h": [1, 20]}}
    hs = HB.rhythm_hours(days)
    assert hs == {0, 1, 2} and 13 not in hs and 20 not in hs


# ------------------------------------------------------------------ B26
def test_b26_habituated_backup_weighs_nothing_inside_and_full_outside():
    rig = Rig()
    rig.nights(8, training=True)                   # 8 trusted warm-up nights
    rig.tick()                                     # first live tick: evidence reset
    assert rig.one(1) == pytest.approx(0.0, abs=1e-9)          # the usual backup
    assert verdicts(rig.store)[-1] == HB.IN
    rig2 = Rig()
    rig2.nights(8, training=True)
    rig2.tick()
    assert rig2.one(14) == pytest.approx(30.0, rel=1e-3)       # a new time: full weight
    st = rig2.risk._states[rig2.store][(S, E)]
    assert "outside habit" in st.detail["lib4:bulk_upload"]
    rig3 = Rig()
    rig3.nights(8, training=True)
    rig3.tick()
    assert rig3.one(1, peers=(BACKUP, "203.0.113.9")) == pytest.approx(30.0, rel=1e-3)
    rig4 = Rig()
    rig4.nights(8, training=True)
    rig4.tick()
    assert rig4.one(1, up=3e9) == pytest.approx(30.0, rel=1e-3)   # 5x the learnt volume


def test_b26_fewer_than_five_days_and_critical_never_habituate():
    rig = Rig()
    rig.nights(4, training=True)
    rig.tick()
    assert rig.one(1) == pytest.approx(30.0, rel=1e-3)
    assert verdicts(rig.store)[-1] == HB.LEARNING
    rig = Rig()
    rig.nights(8, training=True, sev=Severity.CRITICAL)
    rig.tick()
    assert rig.one(1, sev=Severity.CRITICAL) == pytest.approx(50.0, rel=1e-3)
    assert "sigs" not in HB.get(rig.store, S, E)


def test_b26_quarantined_days_are_not_trusted_history():
    rig = Rig()
    rig.tick(training=True)
    rig.tick()                                     # live from here

    def q(st, t, m):                               # B28 plays: quarantined throughout
        st.add_vec(S, E, MG.QUARANTINE, t, [1.0], window_s=int(DT))
        if m:
            put_match(st, t)
    for _ in range(6 * 24):
        m = _hour(rig.t + DT) == 1
        rig.tick(lambda st, t, m=m: q(st, t, m))
    rec = HB.get(rig.store, S, E)["sigs"]["bulk_upload"]
    assert rec["days"] == {} and set(verdicts(rig.store)) == {HB.LEARNING}


def test_b26_live_out_of_envelope_matches_never_widen_it_and_accept_restarts():
    rig = Rig()
    rig.nights(8, training=True)
    rig.tick()
    rig.nights(4, training=False, hour=14)          # trusted but outside every night
    assert set(verdicts(rig.store)[-4:]) == {HB.OUT}
    assert 14 not in HB.rhythm_hours(HB.get(rig.store, S, E)["sigs"]["bulk_upload"]["days"])
    # a governor ACCEPT (model.control version bump) restarts the history
    rig.store.put_model(S, E, MG.CONTROL, {"version": 1}, version=1)
    rig.nights(1, training=False, hour=14)
    rec = HB.get(rig.store, S, E)["sigs"]["bulk_upload"]
    assert len(rec["days"]) == 1 and verdicts(rig.store)[-1] == HB.LEARNING


# ------------------------------------------------------------------ B28
def test_b28_training_trust_kept_for_a_habituated_high_match_only():
    rig = Rig(governor=True)
    trust = {}

    def night(n):
        for _ in range(n * 24):
            nxt = rig.t + DT
            is_m = _hour(nxt) == 1
            rig.tick((lambda st, t: put_match(st, t)) if is_m else None, training=True)
            if _hour(rig.t - DT) == 1:             # the tick after the match reads it
                trust.setdefault("t", []).append(
                    float(rig.store.vec_at(S, E, MG.TRUST, rig.t)[0]))
    night(8)
    got = trust["t"]
    # learning nights: the HIGH match zeroes the warm-up trust; once the
    # backup is habituated (>= 5 trusted nights) it no longer does
    assert got[:5] == [0.0] * 5 and all(v == 1.0 for v in got[5:])
    # outside the envelope (a new hour) it zeroes the trust again
    rig.until_hour(14, training=True)
    rig.tick(lambda st, t: put_match(st, t), training=True)
    rig.tick(training=True)
    assert float(rig.store.vec_at(S, E, MG.TRUST, rig.t)[0]) == 0.0


def test_b28_habitual_high_helper_reads_the_verdict():
    store = make_store()
    m = SignatureMatch(system=S, entity=E, ts=T0, signature_id="bulk_upload", label="",
                       category="transfer", confidence=1.0, severity=Severity.HIGH)
    assert not GV._habitual_high(store, S, E, m)
    store.put_model(S, E, HB.MODEL, {"sigs": {"bulk_upload": {"v": {f"{T0:.3f}": HB.IN}}}})
    assert GV._habitual_high(store, S, E, m)
    m.severity = Severity.CRITICAL
    assert not GV._habitual_high(store, S, E, m)
