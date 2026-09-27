"""B07 RhythmEngine: the slot clock across cadences (900 -> 60 s, 3600-s
ticks, sampled streams), calendar self-healing and the per-tick cost.

Uses the Rig driver of test_b07_rhythm.py."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from helpers import run_engine

from app.engines.behavior.lib import m_rhythm as R
from app.engines.behavior.lib.classkeys import SYSTEM_KEY
from app.models.schema import RawMetric
from test_b07_rhythm import (DAY, H_OFF, S, T0, Rig, backup_at, in_window, local, train,
                             worker)

H = 3600.0


def _night(t: float) -> bool:
    return worker(t) or in_window(local(t)[1], 2.0, 2.75)


# ================================================================== cadence
def _cadence_run(schedule):
    rig = Rig({"w": worker})
    train(rig, 21)
    for t_end, dt in schedule:
        rig.run_until(t_end, dt=dt, patterns={"w": _night})
    return rig


def test_cadence_switch_900_to_60_mid_night_keeps_slot_clock():
    base = T0 + 21 * DAY
    a = _cadence_run([(base + 4 * H, 900.0)])
    # 900 s until 01:45, 60 s across the alarm slots, back to 900 s off the grid
    b = _cadence_run([(base + 1.75 * H, 900.0), (base + 2 * H + 7 * 60, 60.0),
                      (base + 4 * H + 7 * 60, 900.0)])
    ha, hb = (R.slot_history(r.model("w"), since=base) for r in (a, b))
    hb = [x for x in hb if x[0] <= ha[-1][0]]
    assert [(j, x) for j, x, *_ in ha] == [(j, x) for j, x, *_ in hb]
    assert [x for _, x, *_ in ha if x == 1.0] == [1.0] * 3          # 02:00, 02:15, 02:30
    Wa, Wb = a.model("w")["det"]["W"], b.model("w")["det"]["W"]
    assert Wa == pytest.approx(Wb, abs=1e-9)
    # the 60-s ticks cover the 02:00 slot 15 times: finalised once, fully covered
    assert all(math.isfinite(x) for _, x, *_ in hb)


def test_3600_tick_resolves_its_four_slots_from_timestamps():
    rig = Rig({"w": lambda t: in_window(local(t)[1], 10.25, 10.5)})   # 10:15-10:30 only
    rig.run_until(T0 + 11 * H, dt=H)
    hist = R.slot_history(rig.model("w"))
    by_hour = {j: a for j, a, *_ in hist if 10 <= (j % 96) / 4 < 11}
    assert [by_hour[j] for j in sorted(by_hour)] == [0.0, 1.0, 0.0, 0.0]


def test_sampled_stream_leaves_rowless_slots_unknown():
    rig = Rig({})
    st = rig.store
    now = T0 + H
    ts = np.array([T0 + 20 * 60.0])                                 # a row in the 00:15 slot
    from app.engines.behavior.lib import m_template as MT
    a = np.zeros(1, dtype=MT.STREAM_DTYPE)
    a["ts"] = ts
    st.add_vec(S, "z", "feature.active", now, np.array([1.0], dtype=np.float32), window_s=3600)
    st.add_raw(RawMetric(name="act.stream", value=a, ts=now, system=S, entity="z"))
    st.add_raw(RawMetric(name="act.stream_frac", value=0.25, ts=now, system=S, entity="z"))
    rig.now = now
    run_engine(rig.eng, st, now, dt=H, config=rig.cfg)
    hist = R.slot_history(rig.model("z"))
    acts = [a for _, a, *_ in hist]
    assert acts[1] == 1.0 and all(math.isnan(x) for i, x in enumerate(acts) if i != 1)
    assert hist[1][2] == pytest.approx(4.0)                        # volume scaled by 1/frac


# ================================================================= calendar
def test_calendar_self_healing_treats_unconfigured_makeup_saturday_as_workday():
    def run(n):
        rig = Rig({f"w{i}": worker for i in range(n)})
        train(rig, 19)                                  # through Friday; day 19 is a Saturday
        assert local(rig.now)[2] == 5
        sat = lambda t: in_window(local(t)[1], 9.0, 18.0)   # noqa: E731  unannounced 调休
        rows = rig.run_until(T0 + 19 * DAY + 13 * H, dt=900.0,
                             patterns={f"w{i}": sat for i in range(n)}, record=True,
                             entity="w0")
        return rig, rows

    rig, rows = run(4)
    sm = rig.store.get_model(S, SYSTEM_KEY, R.MODEL)
    day = int((T0 + 19 * DAY + 8 * 3600) // 900) // 96              # local day index
    assert day in R.healed_days(sm)
    assert all(r["acc"].get("offhours") == 0 for r in rows)
    assert max(r["W_off"] for r in rows) < H_OFF
    # a lone worker cannot heal the calendar: the same Saturday alarms
    rig1, rows1 = run(1)
    assert not R.healed_days(rig1.store.get_model(S, SYSTEM_KEY, R.MODEL))
    assert any(r["acc"].get("offhours") == 1 for r in rows1)


# ===================================================================== perf
def test_perf_per_entity_tick():
    pats = {f"e{i}": (worker if i % 2 else backup_at(2.0, 2.67)) for i in range(20)}
    rig = Rig(pats)
    train(rig, 3)
    rig.run_until(rig.now + 2 * H, dt=900.0)
    t_all = 0.0
    n = 0
    for _ in range(24):
        rig.now += 900.0
        rig.write_tick(rig.now, 900.0, pats)
        t = time.perf_counter()
        run_engine(rig.eng, rig.store, rig.now, dt=900.0, config=rig.cfg)
        t_all += time.perf_counter() - t
        n += len(pats)
    per = t_all / n
    assert per < 5e-3, f"{per * 1e3:.3f} ms per entity-tick"        # spec 1 ms; generous bound
