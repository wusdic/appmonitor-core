"""Tick-mode golden fingerprint (docs/lib3/cadence.md §12 M0, §16 risk 8).

`grain_mode = 'tick'` must reproduce lib-3 v2 exactly while the cadence
migration lands engine by engine. This test replays a short mini-pack run
(48 x 3600 s warm-up + 48 x 900 s live, the mini population and scenarios)
in tick mode and compares it with the fingerprint recorded BEFORE any engine
was changed (tests/data/tick_mode_golden.json.gz):

  * behavior.p of the 31 v2 detectors (the appended Q columns stay NaN in
    tick mode) as -log10 p, per entity (real and class) and live tick;
  * behavior.e_day, the alarm path / severity per entity and tick;
  * the incidents (entity, opened, severity).

Values are compared at 1e-6 relative (-log10 p) so the test is robust to
the float32 ring rounding but catches any semantic change. Regenerate only
for a deliberate v2 change: `python -c "import tests.test_tick_mode_golden as g; g.record()"`.
"""
from __future__ import annotations

import datetime as _dt
import gzip
import json
import math
import os

import numpy as np
import pytest

from app.eval import packs as P
from app.eval.runner import run_pack

GOLDEN = os.path.join(os.path.dirname(__file__), "data", "tick_mode_golden.json.gz")
N_V2 = 31


def golden_pack() -> P.Pack:
    cal = {"holidays": [], "makeup_workdays": []}
    tl = P.Timeline(P.SHANGHAI, cal, _dt.date(2025, 3, 10), P._phases((48, 3600.0), (48, 900.0)))
    scs = [P.t1(tl, tl.tick(40)), P.t6(tl, tl.tick(44)), P.l14(tl, tl.tick(30), "10.20.1.13")]
    scs[1].t_end = tl.end
    pk = P._pack("golden", tl, P.MINI_KEYS, scs, "tick-mode golden (mini population)")
    pk.config["grain_mode"] = "tick"
    return pk


def _nlp(x) -> list:
    a = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = -np.log10(np.clip(a, 1e-300, 1.0))
    return [None if not math.isfinite(float(t)) else round(float(t), 5) for t in v.reshape(-1)]


def fingerprint() -> dict:
    res = run_pack(golden_pack(), seed=0, strict=True, on_error="record")
    assert res.exceptions == [], res.exceptions[:2]
    out = {"series": {}, "incidents": []}
    for k in sorted(res.series):
        s = res.series[k]
        p = np.asarray(s["p"], dtype=np.float64)[:, :N_V2]
        out["series"][k] = {
            "ts": [float(t) for t in s["ts"]],
            "p": [_nlp(row) for row in p],
            "e_day": _nlp(s["e_day"]),
            "alarm": [f"{a}:{b}" for a, b in zip(s["alarm_path"], s["alarm_sev"])],
        }
    for inc in res.incidents:
        out["incidents"].append([inc.get("system"), inc.get("entity"),
                                 float(inc.get("opened") or 0.0), str(inc.get("severity"))])
    return out


def record() -> None:  # pragma: no cover - maintenance helper
    fp = fingerprint()
    with gzip.open(GOLDEN, "wt", encoding="utf-8") as f:
        json.dump(fp, f)


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b)) + 2e-5


@pytest.mark.skipif(not os.path.exists(GOLDEN), reason="golden fingerprint not recorded")
def test_tick_mode_reproduces_v2_fingerprint():
    with gzip.open(GOLDEN, "rt", encoding="utf-8") as f:
        want = json.load(f)
    got = fingerprint()
    assert sorted(got["series"]) == sorted(want["series"])
    for k, w in want["series"].items():
        g = got["series"][k]
        assert g["ts"] == w["ts"], k
        assert g["alarm"] == w["alarm"], k
        for i, (rw, rg) in enumerate(zip(w["p"], g["p"])):
            bad = [j for j, (a, b) in enumerate(zip(rw, rg)) if not _close(a, b)]
            assert not bad, (k, i, bad[:4], [rw[j] for j in bad[:4]], [rg[j] for j in bad[:4]])
        bad = [i for i, (a, b) in enumerate(zip(w["e_day"], g["e_day"])) if not _close(a, b)]
        assert not bad, (k, "e_day", bad[:4])
    assert got["incidents"] == want["incidents"]
