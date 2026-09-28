"""B07 RhythmEngine: the spec unit tests (engines.md B07 (a)-(e)).

The `Rig` driver (a store with feature.active + act.stream per tick, as B01
and R2 write them) is shared with test_b07_rhythm_{edges,clock,shift}.py. Training runs at
3600-s ticks (the slot clock resolves each hour into its 4 slots from the
stream timestamps) to keep the files fast; the scored days run at 900 s or
60 s."""
from __future__ import annotations

import datetime as _dt
import math
from typing import Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from helpers import make_store, run_engine

from app.engines.behavior.lib import emit
from app.engines.behavior.lib import m_rhythm as R
from app.engines.behavior.lib import m_template as MT
from app.engines.behavior.lib import seq as SQ
from app.engines.behavior.rhythm import SERIES, RhythmEngine
from app.models.schema import RawMetric

TZ = "Asia/Shanghai"
S = "erp"
DAY = 86400.0
T0 = _dt.datetime(2026, 1, 5, tzinfo=ZoneInfo(TZ)).timestamp()   # Monday 00:00 local
EV_STEP = 300.0                  # one event every 5 min while "on"
EV_OFF = 17.0                    # events at hh:m5:17, never on a tick boundary
H_OFF = R.H_OFF
H_SIL = R.H_SIL

# 4 weeks: a daily job lands in the workday cells only 5 days in 7 and the
# 3-member system prior (strength 2) pulls toward the pooled rate, so 3 weeks
# leave p_hat ~ 0.949 (< 0.95, ineligible); 4 weeks give ~ 0.958.
SCHED_DAYS = 28
Pattern = Callable[[float], bool]


def local(t: float):
    """(day index since T0, local hour as float, dow Mon=0)."""
    L = t - T0
    d = int(L // DAY)
    return d, (L - d * DAY) / 3600.0, d % 7


def in_window(h: float, h0: float, h1: float) -> bool:
    return h0 <= h < h1


def worker(t: float) -> bool:
    _, h, dow = local(t)
    return dow < 5 and in_window(h, 9.0, 18.0)


def backup_at(h0: float, h1: float) -> Pattern:
    return lambda t: in_window(local(t)[1], h0, h1)


class Rig:
    """One system, entities driven by activity patterns (t -> bool)."""

    def __init__(self, patterns: Dict[str, Pattern], config: Optional[dict] = None,
                 t0: float = T0) -> None:
        self.store = make_store()
        self.eng = RhythmEngine()
        self.patterns = dict(patterns)
        self.now = t0
        self.cfg = {"tz": TZ}
        self.cfg.update(config or {})
        self.rows: List[dict] = []

    def events(self, fn: Pattern, t0: float, t1: float) -> List[float]:
        k = math.ceil((t0 - T0 - EV_OFF) / EV_STEP)
        out = []
        t = T0 + EV_OFF + k * EV_STEP
        while t < t1:
            if t >= t0 and fn(t):
                out.append(t)
            t += EV_STEP
        return out

    def write_tick(self, now: float, dt: float, patterns: Dict[str, Pattern]) -> None:
        st = self.store
        for e, fn in patterns.items():
            ev = self.events(fn, now - dt, now)
            st.add_vec(S, e, "feature.active", now,
                       np.array([1.0 if ev else 0.0], dtype=np.float32), window_s=int(dt))
            if ev:
                a = np.zeros(len(ev), dtype=MT.STREAM_DTYPE)
                a["ts"] = ev
                st.add_raw(RawMetric(name="act.stream", value=a, ts=now, system=S, entity=e))
                st.add_raw(RawMetric(name="act.events", value=float(len(ev)), ts=now,
                                     system=S, entity=e))
            else:
                st.register_entity(S, e)

    def step(self, dt: float = 900.0, training: bool = False,
             patterns: Optional[Dict[str, Pattern]] = None) -> float:
        self.now += dt
        self.write_tick(self.now, dt, patterns or self.patterns)
        run_engine(self.eng, self.store, self.now, training=training, dt=dt, config=self.cfg)
        return self.now

    def run_until(self, t_end: float, dt: float = 900.0, training: bool = False,
                  patterns: Optional[Dict[str, Pattern]] = None, record: bool = False,
                  entity: Optional[str] = None) -> List[dict]:
        out = []
        while self.now + dt <= t_end + 1e-6:
            self.step(dt, training, patterns)
            if record:
                out.append(self.snap(entity))
        return out

    def snap(self, e: str) -> dict:
        st, now = self.store, self.now
        m = st.latest_derived(S, e, SERIES)
        v = dict(m.value) if m is not None and m.ts == now else {}
        v["ts"] = now
        v["score"] = emit.read_row(st, S, e, emit.SCORE, now)
        v["pm"] = emit.read_row(st, S, e, emit.PM, now)
        v["acc"] = emit.read_dict(st, S, e, emit.ACC_ALARM, now)
        v["axes"] = emit.read_dict(st, S, e, emit.AXES, now)
        return v

    def model(self, e: str) -> dict:
        return self.store.get_model(S, e, R.MODEL)

    def events_of(self, e: str, kind: str = "schedule_shift"):
        return self.store.events(S, e, kinds=(kind,), limit=100)


def train(rig: Rig, days: int, dt: float = 3600.0) -> None:
    rig.run_until(T0 + days * DAY, dt=dt, training=True)


# ======================================================================= (a)
def _night_run(dt_night: float) -> List[dict]:
    """Worker 09-18 on workdays: 21 training days (3600 s), a normal Monday
    live at 900 s, then Tuesday 00:00-04:00 at `dt_night` with activity at
    02:00-02:45 (3 slots)."""
    rig = Rig({"w": worker})
    train(rig, 21)
    normal = rig.run_until(T0 + 22 * DAY, dt=900.0, record=True, entity="w")
    night = lambda t: worker(t) or in_window(local(t)[1], 2.0, 2.75)   # noqa: E731
    rows = rig.run_until(T0 + 22 * DAY + 4 * 3600, dt=dt_night, patterns={"w": night},
                         record=True, entity="w")
    return normal, rows, rig


def _first_alarm(rows: List[dict]) -> Optional[dict]:
    for r in rows:
        if r.get("acc", {}).get("offhours") == 1:
            return r
    return None


def test_a_offhours_third_active_slot_alarms_same_slot_and_W_at_60_and_900():
    normal900, rows900, rig900 = _night_run(900.0)
    _, rows60, rig60 = _night_run(60.0)
    # the model: quiet workday night cell after 20 days of silent nights
    m = rig900.model("w")
    p02 = R.p_cell(m, R.cell48(2, 0))
    assert p02 <= 0.02
    assert R.det_p(p02) == pytest.approx(0.02)
    # a normal workday gives W = 0 and no alarm
    assert normal900 and all(r["W_off"] == 0.0 for r in normal900)
    assert all(r["acc"].get("offhours") == 0 for r in normal900)
    # 900 s: slots 02:00, 02:15 below h, the 3rd alarms with W = 3 x 4.64 = 13.9
    a900 = _first_alarm(rows900)
    assert a900 is not None
    W3 = 3.0 * math.log2(0.5 / 0.02)
    assert a900["W_off"] == pytest.approx(W3, abs=1e-6)
    assert a900["W_off"] >= H_OFF == pytest.approx(13.73, abs=0.01)
    before = [r for r in rows900 if r["ts"] < a900["ts"]]
    assert max(r["W_off"] for r in before) == pytest.approx(2.0 * math.log2(0.5 / 0.02), abs=1e-6)
    assert max(r["W_off"] for r in before) < H_OFF
    slot3 = a900["w_slot"]
    _, hh, _ = local(slot3 * 900.0 - 8 * 3600 + 1)      # local slot index -> local time
    assert hh == pytest.approx(2.5, abs=1e-3)            # the 02:30 slot
    assert a900["score"]["offhours"] == pytest.approx(a900["W_off"], rel=1e-5)
    assert a900["pm"]["offhours"] == pytest.approx(2.0 ** -a900["W_off"], rel=1e-4)
    assert a900["axes"].get("offhours") == ["temporal"]
    # 60 s: the alarm is raised provisionally inside the same wall-clock slot, same W
    a60 = _first_alarm(rows60)
    assert a60 is not None
    assert a60["w_slot"] == slot3
    assert a60["W_off"] == pytest.approx(a900["W_off"], abs=1e-9)
    assert a60["ts"] < a900["ts"]                        # 02:31 vs the 02:45 tick
    # both cadences end the night in the same state (slot clock, not ticks)
    assert rows60[-1]["W_off"] == pytest.approx(rows900[-1]["W_off"], abs=1e-9)
    assert rig60.model("w")["det"]["W"] == pytest.approx(rig900.model("w")["det"]["W"], abs=1e-9)


# =================================================================== (b)-(e)
def human(t: float) -> bool:
    """Irregular daytime activity every day (weekends too) plus the backup
    slots 02:00-02:40, so only the automation gate differs from `bk`."""
    d, h, _ = local(t)
    if in_window(h, 2.0, 2.67):
        return True
    if not in_window(h, 7.0, 23.0):
        return False
    k = int((t - T0) // 900)
    return (k * 2654435761 % 1000) < 550                  # deterministic ~55 % of slots


@pytest.fixture(scope="module")
def sched():
    """bk: backup 02:00-02:40 daily; hu: human rhythm with the same slots;
    sh: backup 01:00-01:40 daily. SCHED_DAYS training days, then one night
    live at 900 s: bk and hu skip their night, sh moves to 03:00-03:40 (same
    volume)."""
    rig = Rig({"bk": backup_at(2.0, 2.67), "hu": human, "sh": backup_at(1.0, 1.67)})
    train(rig, SCHED_DAYS)
    skip = {
        "bk": lambda t: False,
        "hu": lambda t: human(t) and not in_window(local(t)[1], 0.0, 6.0),
        "sh": backup_at(3.0, 3.67),
    }
    out = {e: [] for e in skip}
    while rig.now + 900.0 <= T0 + SCHED_DAYS * DAY + 6 * 3600 + 1e-6:
        rig.step(900.0, patterns=skip)
        for e in skip:
            out[e].append(rig.snap(e))
    return rig, out


def test_b_backup_skip_gives_silence_p_le_1e4(sched):
    rig, out = sched
    m = rig.model("bk")
    assert m["machine_like"] and R.automation_index(m) >= R.AUTO_ENTER
    rows = out["bk"]
    # p_hat of the three skipped slots as scored (the rows report the slot
    # just finalised; the model has since learned the miss)
    sil = [r for r in rows if r["s_sil"] > 0.0]
    assert [r["p_expected"] >= R.P_SIL_MIN for r in sil[:3]] == [True] * 3
    assert len({r["s_sil"] for r in sil}) == 3                # three increments, then flat
    s_max = max(r["s_sil"] for r in rows)
    assert math.exp(-s_max) <= 1e-4
    assert s_max >= H_SIL
    alarm = [r for r in rows if r["acc"].get("silence") == 1]
    assert alarm
    first = alarm[0]
    assert first["pm"]["silence"] <= 1e-4
    assert first["axes"].get("silence") == ["temporal"]
    # accrues only after the 3 scheduled slots have passed silently
    _, hh, _ = local(first["ts"] - 1)
    assert 2.5 <= hh < 3.0
    assert all(r["acc"].get("offhours") == 0 for r in rows)


def test_c_same_skip_for_a_human_rhythm_is_ineligible(sched):
    rig, out = sched
    m = rig.model("hu")
    # spec change (W7): machine_like is the automation index (m_rhythm), not
    # entropy <= 0.8; this every-day 07-23 coin-flip rhythm fails its
    # regularity requirement (the old rule failed it on entropy)
    comps = R.automation_components(m)
    assert comps["regular"] < R.REGULAR_MIN and not m["machine_like"]
    assert R.automation_components(rig.model("bk"))["regular"] >= R.REGULAR_MIN
    # the backup slots alone would qualify (p >= 0.95): only the automation gate differs
    assert min(R.p_cell(m, R.cell48(2, q)) for q in range(3)) >= R.P_SIL_MIN
    assert not R.silence_eligible(0.99, m["machine_like"])
    rows = out["hu"]
    assert all("silence" not in r["score"] for r in rows)            # NaN = unscored
    assert all("silence" not in r["pm"] for r in rows)
    assert all(r["s_sil"] is None for r in rows)
    assert all(r["acc"].get("silence") == 0 for r in rows)
    assert all(math.isfinite(r["score"]["offhours"]) for r in rows)


def test_d_offhours_bins_at_03_and_05():
    # p = 0.3 is still an off-hours bin: finite, p1 = min(.95, 1.5) = .95
    w1 = R.offhours_step(0.0, 1.0, 0.3)
    assert math.isfinite(w1) and w1 == pytest.approx(math.log2(0.95 / 0.3))
    w0 = R.offhours_step(5.0, 0.0, 0.3)
    assert math.isfinite(w0) and w0 == pytest.approx(5.0 + math.log2(0.05 / 0.7))
    # p = 0.5 contributes exactly 0 (never p1 > 1, never NaN)
    for a in (0.0, 1.0):
        for W in (0.0, 3.25):
            assert R.offhours_step(W, a, 0.5) == W
    assert math.isnan(SQ.rhythm_p1(0.5))
    # NaN activity (unobserved slot) and NaN p leave W unchanged
    assert R.offhours_step(2.0, math.nan, 0.01) == 2.0
    assert R.offhours_step(2.0, 1.0, math.nan) == 2.0
    # the p = 0.02 floor: an active slot is worth 4.64 bits, 3 slots cross h
    assert R.offhours_step(0.0, 1.0, 0.001) == pytest.approx(math.log2(0.5 / 0.02))
    W = 0.0
    for _ in range(3):
        W = R.offhours_step(W, 1.0, 0.02)
    assert W == pytest.approx(13.93, abs=0.01) and W >= H_OFF


def test_e_schedule_shift_emits_low_event_and_caps_temporal(sched):
    rig, out = sched
    evs = rig.events_of("sh")
    assert len(evs) == 1
    ev = evs[0]
    assert ev.kind == "schedule_shift"
    assert ev.severity.value == "low"
    assert ev.axes == ["temporal"]
    assert ev.extra["from"] == "01:00" and ev.extra["to"] == "03:00"
    assert ev.extra["shift_h"] == pytest.approx(2.0)
    assert ev.extra["cap"] == "low" and ev.extra["shift_explained"] is True
    assert 0.5 <= ev.extra["vol_ratio"] <= 2.0
    rows = out["sh"]
    after = [r for r in rows if r["ts"] >= ev.ts]
    assert after and all(r["shift_explained"] == 1 for r in after)
    # the silence of the old window was explained: evidence reset, no latched alarm
    assert all(r["acc"].get("silence") == 0 and r["acc"].get("offhours") == 0 for r in after)
    assert all(r["W_off"] == 0.0 or r["W_off"] < H_OFF for r in after)
    assert rig.model("sh")["shifts"][-1]["to"] == "03:00"
    prof = rig.store.profile(S, "sh")
    assert prof.extra["rhythm"]["shift_explained"] is True
    # the skipping backup and the human got no shift
    assert not rig.events_of("bk") and not rig.events_of("hu")
