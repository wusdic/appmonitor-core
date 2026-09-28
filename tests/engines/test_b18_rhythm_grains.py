"""B18 class_rhythm in canonical grain mode (evaluator round 3).

(1) Per-slot activity comes from the members' act.slot_events, so the slot
    observations (a active of m present) are the same at 900 s and 3600 s.
    The v2 path counted a member active in a tick's slot when it was active
    anywhere in the tick, so a 3600-s warm-up taught the HOUR's activity as
    one slot's (4 members each active in one different quarter of the hour:
    a = 4 of 4 at 3600 s, a = 1 of 4 per slot at 900 s), and the 900-s live
    slots read as a class-wide drop (pack B: erp-prod class:r3 S_lo latched
    Sat 20:15 -> Tue 23:15).
(2) A slot whose own bin (local hour x day type) holds fewer than 3 decayed
    slots (most of one observed day) is not scored: an empty bin fell back to the pooled prior,
    i.e. to the other bins (smoke run at Mon 09:13 after a weekend 900-s
    phase: the human class scored its Monday-morning slots against weekend
    activity, S_hi = 21 by the first live tick, HIGH temporal incident).
"""
from __future__ import annotations

import numpy as np
import pytest

from helpers import make_store, run_engine

from app.engines.behavior import class_monitor as CM
from app.engines.behavior.class_monitor import ClassMonitorEngine, new_model
from app.models.schema import RawMetric

from test_b18_class_monitor import CK, S, T, member_tick, nat_row, put_classes

IPS = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]
CANON = {"grain_mode": "canonical"}


def _slot_obs(dt: float, hours: int = 6):
    """Member i has one event in quarter i of every hour; returns the
    (bin48, a, m) of every slot B18 closes."""
    st, eng, rng = make_store(), ClassMonitorEngine(), np.random.default_rng(0)
    put_classes(st, {"r1": IPS})
    seen = []
    orig = eng._score_slot

    def spy(s, ck, model, b, a, m, slot_id, now):
        seen.append((b, a, m))
        return orig(s, ck, model, b, a, m, slot_id, now)
    eng._score_slot = spy
    n = int(hours * 3600 / dt)
    for k in range(1, n + 1):
        now = T + k * dt
        for i, ip in enumerate(IPS):
            # the quarter-hour slots this tick covers in which member i has its event
            ev = {}
            q = now - dt
            while q < now - 1e-6:
                if int((q % 3600) // 900) == i:
                    ev[float(q - q % 900)] = 3.0
                q += 900.0
            member_tick(st, ip, now, nat_row(rng), active=bool(ev), dt=dt)
            if ev:
                st.add_raw(RawMetric(name="act.slot_events", value=ev, ts=now, system=S,
                                     entity=ip))
        run_engine(eng, st, now, training=True, dt=dt, config=CANON)
    return seen


def test_slot_activity_is_the_same_at_900_and_3600():
    a900, a3600 = _slot_obs(900.0), _slot_obs(3600.0)
    assert a900 and a900 == a3600
    assert {(a, m) for _b, a, m in a900} == {(1, 4)}


def _model_with_pool(bin_w: float):
    """A class model whose pooled bin has 40 slots at 10 % activity and whose
    bin 9 (workday 09:00) holds `bin_w` slots at 80 %."""
    m = new_model("role", canon=True)
    rh = m["aux"]["rh"]
    for _ in range(40):
        CM._dec_fold(rh, T, CM.RHYTHM_HL_S, 48, np.array([1.0, 0.4, 4.0, 0.04]), 1.0)
    for _ in range(int(bin_w)):
        c = np.array([1.0, 3.2, 4.0, 3.2 * 3.2 / 4.0])
        CM._dec_fold(rh, T, CM.RHYTHM_HL_S, 9, c, 1.0)
        CM._dec_fold(rh, T, CM.RHYTHM_HL_S, 48, c, 1.0)
    return m


@pytest.mark.parametrize("bin_w,scored", [(0, False), (2, False), (4, True), (12, True)])
def test_a_slot_of_an_unobserved_bin_is_not_scored(bin_w, scored):
    eng = ClassMonitorEngine()
    model = _model_with_pool(bin_w)
    for k in range(8):                       # 4 of 4 members active at 09:00
        obs = eng._score_slot(S, CK, model, 9, 4, 4, 1000 + k, T + 900.0 * (k + 1))
        assert obs == (9, 4, 4)              # learned either way
    run = model["run"]
    assert bool(run["rh_scored"]) is scored
    if not scored:
        assert run["S_hi"] == 0.0 and run["S_lo"] == 0.0
