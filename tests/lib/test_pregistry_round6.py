"""P02 round 6 (cost): the hourly update_types + update_stats pair shares one
distinct count per record (an EpochHLL copy-merge-estimate each) and leaves
every registry as the separate passes did, on the recorded pack O stream."""
from __future__ import annotations

import gzip
import pickle

import pytest

import test_pregistry_batch_equivalence as EQ

from app.engines.behavior.lib import pregistry as RG


@pytest.fixture(scope="module")
def stream():
    with gzip.open(EQ.STREAM, "rb") as f:
        return pickle.load(f)


def _replay(stream, shared: bool):
    regs, last = {}, {}
    for system, meth, args, kw in stream:
        reg = regs.get(system)
        if reg is None:
            reg = regs[system] = RG.AttrRegistry(system)
        if shared and meth in ("update_types", "update_stats") and not kw:
            # P02's hourly pair: update_types(t) then update_stats(t) on one dict
            key = (system, args[0])
            if meth == "update_types" or last.get(system, (None,))[0] != key:
                last[system] = (key, {})
            getattr(reg, meth)(*args, counts=last[system][1])
            continue
        getattr(reg, meth)(*args, **kw)
    return regs


def test_shared_distinct_counts_leave_the_registries_unchanged(stream, monkeypatch):
    n = {"c": 0}
    real = RG._card_count

    def spy(rec, counts):
        if counts is None or id(rec.card) not in counts:
            n["c"] += 1
        return real(rec, counts)
    monkeypatch.setattr(RG, "_card_count", spy)
    ref = _replay(stream, False)
    n_ref, n["c"] = n["c"], 0
    new = _replay(stream, True)
    assert n["c"] < n_ref
    for s in ref:
        a, b = new[s], ref[s]
        assert list(a.records) == list(b.records)
        for nm in b.records:
            assert EQ.canon(a.records[nm]) == EQ.canon(b.records[nm]), (s, nm)
