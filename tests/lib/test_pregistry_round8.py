"""(round 8, harness owner) P02 cost with the number of attributes."""
from __future__ import annotations

import numpy as np

from app.engines.behavior.lib import pregistry as RG


def _reg(n_attr: int = 6):
    reg = RG.AttrRegistry("oa")
    r = np.random.default_rng(0)
    for k in range(n_attr):
        reg.observe(f"meta.f{k:03d}", [f"v{int(x)}" for x in r.integers(0, 5, 40)], 1000.0, 1.0)
    return reg


def test_update_stats_recomputes_only_attributes_that_changed(monkeypatch):
    """An attribute nothing was added to since the last pass keeps its entropy
    and stability (they read stored counts only): the hourly pass costs the
    attributes observed in the hour, not every registered one."""
    reg = _reg()
    calls = []
    real = RG._distributions
    monkeypatch.setattr(RG, "_distributions", lambda *a, **k: calls.append(1) or real(*a, **k))
    reg.update_stats(4600.0)
    assert len(calls) == 6
    before = {n: (r.entropy, r.stability) for n, r in reg.records.items()}
    calls.clear()
    reg.update_stats(8200.0)                          # an hour later, nothing observed
    assert calls == []
    reg.observe("meta.f002", ["v1", "v9"], 8300.0, 1.0)
    reg.set_role("meta.f004", "dropped")
    reg.update_stats(11800.0)
    assert len(calls) == 1                            # f002 (f004's top was released: empty)
    for n, r in reg.records.items():
        if n not in ("meta.f002", "meta.f004"):
            assert (r.entropy, r.stability) == before[n]


def test_skipped_statistics_equal_a_full_recomputation():
    reg = _reg()
    r = np.random.default_rng(1)
    for h in range(1, 30):
        t = 1000.0 + 3600.0 * h
        for k in r.choice(6, 2, replace=False):
            reg.observe(f"meta.f{k:03d}", [f"v{int(x)}" for x in r.integers(0, 7, 10)], t - 10.0, 1.0)
        reg.update_stats(t)                           # the skipping pass
        snap = {n: (x.entropy, x.stability) for n, x in reg.records.items()}
        for rec in reg.records.values():
            rec.stats_dirty = True
        reg.update_stats(t)                           # every record recomputed
        assert {n: (x.entropy, x.stability) for n, x in reg.records.items()} == snap
