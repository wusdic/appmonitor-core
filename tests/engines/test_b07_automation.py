"""B07 machine_like: the calibrated automation index (W7 tuning).

The v2 rule (normalised 168-bin entropy <= 0.8) held for an office worker
(09-18 on workdays: 45 of 168 hours, entropy ~0.74) and for no 24/7 client
(entropy ~1), so silence was scored for people's days off and never for a
stopped service. m_rhythm.automation_index combines the off-hours activity
ratio, the non-workday / workday ratio, presence regularity and B02's A.
The generator-level validation (37 personas x 2 seeds, 10 clean days at
3600 s: humans <= 0.34, machines >= 0.73) is in integration.md §8.2.
"""
from __future__ import annotations

import math

from test_b07_rhythm import DAY, T0, Rig, backup_at, train, worker

from app.engines.behavior.lib import m_rhythm as R


def poller(t: float) -> bool:
    return True                                     # 24/7 (every 5 min, every day)


def test_office_worker_is_human_and_a_24_7_poller_is_a_machine():
    rig = Rig({"w": worker, "p": poller, "bk": backup_at(2.0, 2.67)})
    train(rig, 14)
    w, p, bk = rig.model("w"), rig.model("p"), rig.model("bk")
    # the old rule had both wrong
    assert R.entropy168(w) <= R.ENTROPY_MAX and R.entropy168(p) > R.ENTROPY_MAX
    cw, cp = R.automation_components(w), R.automation_components(p)
    assert cw["offhours"] < 0.05 and cw["week"] < 0.05
    assert cp["offhours"] > 0.95 and cp["week"] > 0.95 and cp["regular"] > 0.95
    assert R.automation_index(w) < R.AUTO_LEAVE and not w["machine_like"]
    assert R.automation_index(p) >= R.AUTO_ENTER and p["machine_like"]
    assert R.automation_index(bk) >= R.AUTO_ENTER and bk["machine_like"]
    # B02's A enters the mean: a human-level A keeps the worker human
    assert R.automation_index(w, 0.2) < R.AUTO_LEAVE
    # descriptors carry the index for the portrait
    assert R.descriptors(p)["machine_like"] is True


def test_a_stopped_24_7_poller_scores_silence_and_a_worker_day_off_does_not():
    rig = Rig({"w": worker, "p": poller})
    train(rig, 14)
    t_off = T0 + 14 * DAY                           # a Monday: the worker stays home,
    stop = {"w": lambda t: False, "p": lambda t: False}      # the poller stops
    rows_p = rig.run_until(t_off + 12 * 3600, dt=900.0, patterns=stop, record=True,
                           entity="p")
    sil_p = [r for r in rows_p if r["score"].get("silence") is not None]
    assert sil_p and max(r["s_sil"] for r in sil_p) >= R.H_SIL
    assert any(r["acc"].get("silence") == 1 for r in rows_p)
    rows_w = [rig.snap("w")]
    assert all("silence" not in r["score"] for r in rows_w)          # unscored


def test_hysteresis_and_regularity_gate():
    assert R.machine_decision(0.5, previous=True) is True
    assert R.machine_decision(0.5, previous=False) is False
    assert R.machine_decision(0.39, previous=True) is False
    assert R.machine_decision(0.61, previous=False) is True
    assert R.machine_decision(0.9, previous=True, regular=0.2) is False
    assert R.machine_decision(float("nan"), previous=True) is True
    # a model spanning less than a week identifies nothing
    m = R.new_model("entity")
    assert all(math.isnan(v) for v in R.automation_components(m).values())
    assert math.isnan(R.automation_index(m)) and not R.machine_like(m)
