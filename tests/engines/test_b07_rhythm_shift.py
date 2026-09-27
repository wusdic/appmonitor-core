"""B07 RhythmEngine: a schedule move is explained before its off-hours
evidence can alarm (spec step 5 / test e), at 900-s and 60-s cadence.

Uses the Rig driver of test_b07_rhythm.py."""
from __future__ import annotations

import pytest

from test_b07_rhythm import DAY, H_OFF, T0, Rig, backup_at, local, train

H = 3600.0


# ========================================================== schedule shift
@pytest.mark.parametrize("dt_live", [900.0, 60.0])
def test_shift_declared_before_the_moved_run_can_alarm(dt_live):
    """A lone 4-slot job (hyperprior only: p_hat < 0.02 at night, 4.64 bits a
    slot) moves 01:00 -> 04:00. Off-hours alone would alarm on the 3rd moved
    slot; while the run may still be the missed window (<= 4 + 1 slots) its
    evidence is held, and the run's end declares the shift (W reset), so the
    explained move never alarms. A run twice as long is no move and alarms
    once it outgrows the tolerance."""
    def night(pattern):
        rig = Rig({"mv": backup_at(1.0, 2.0)})
        train(rig, 28)
        rows = rig.run_until(T0 + 28 * DAY + 7 * H, dt=dt_live, patterns={"mv": pattern},
                             record=True, entity="mv")
        return rig, rows

    rig, rows = night(backup_at(4.0, 5.0))
    evs = rig.events_of("mv")
    assert len(evs) == 1 and evs[0].extra["from"] == "01:00" and evs[0].extra["to"] == "04:00"
    assert local(evs[0].ts)[1] <= 5.25 + 1e-9              # the slot after the run
    assert all(r["acc"].get("offhours") == 0 for r in rows)
    assert max(r["W_off"] for r in rows) < H_OFF
    rig2, rows2 = night(backup_at(4.0, 6.0))                          # twice as long: not a move
    assert not rig2.events_of("mv")
    first = next(r for r in rows2 if r["acc"].get("offhours") == 1)
    assert local(first["ts"])[1] <= 5.5 + 1e-9                     # 6th slot: past 4 + 1
