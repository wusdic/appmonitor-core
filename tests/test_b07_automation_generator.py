"""B07 automation index on the generator population (slow; APPMON_SLOW=1).

10 clean warm-up days at 3600 s over the 37 base personas (+ one live tick):
every human persona (interactive, search, NAT) must be below the leave
threshold and every machine persona (API, integration, health, backup)
above the enter threshold, with B07's own machine_like decision agreeing.
Measured (seeds 0 and 1): humans <= 0.34, machines >= 0.73; the v2 entropy
rule called 22 of 36 human runs and 4 of 38 machine runs machine-like
(integration.md §8.2).
"""
from __future__ import annotations

import datetime as _dt
import os

import pytest

from app.engines.behavior.lib import m_class
from app.engines.behavior.lib import m_rhythm as R
from app.eval import packs as P
from app.eval.runner import run_pack
from app.pipeline.generator import _ARCH, BASE_KEYS

HUMAN = {"interactive", "search", "nat"}


@pytest.mark.skipif(os.environ.get("APPMON_SLOW") != "1", reason="slow: APPMON_SLOW=1")
def test_automation_index_separates_generator_humans_and_machines():
    cal = {"holidays": [], "makeup_workdays": []}
    tl = P.Timeline(P.SHANGHAI, cal, _dt.date(2025, 3, 10), P._phases((240, 3600.0), (1, 3600.0)))
    pk = P._pack("b07-automation", tl, BASE_KEYS, [], "B07 automation index validation")
    res = run_pack(pk, seed=0, strict=True, record_series=False, keep_store=True)
    assert res.exceptions == []
    st = res.store
    hum, mac = [], []
    for s in st.systems():
        for e in st.entities(s):
            m = st.get_model(s, e, R.MODEL)
            arch = _ARCH.get(f"{s}|{e}", ("?", ""))[0]
            if m is None or arch == "?":
                continue
            a = (m_class.assignment(st, s, e) or {}).get("A")
            idx = R.automation_index(m, a)
            (hum if arch in HUMAN else mac).append((f"{s}|{e}", idx, bool(m["machine_like"])))
    assert len(hum) == 18 and len(mac) == 19               # the base population
    assert max(i for _, i, _ in hum) < R.AUTO_LEAVE, hum
    assert min(i for _, i, _ in mac) >= R.AUTO_ENTER, mac
    assert not any(ml for _, _, ml in hum) and all(ml for _, _, ml in mac)
