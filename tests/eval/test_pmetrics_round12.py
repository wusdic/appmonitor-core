"""Round 12 (evaluator, §16.19.12 item 14): the run JSON keeps every failed
precision claim's statement and the incidents PG6's FAR counts (entity, day,
severity, constraint kinds), so a regression traces to a statement or a
constraint without re-running. Information only: no score changes."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from app.eval import packs as P
from app.eval import pmetrics as M
from app.pipeline.generator import TrafficGenerator


@pytest.fixture(scope="module")
def org_run():
    pack = P.get_pack("O")
    g = TrafficGenerator(seed=0, pack=pack)
    for _ in range(pack.n_ticks):
        g.step(900.0)
    return pack, g, M.PTruth(g.ptruth)


def _fake_run(pack, g, events=(), incidents=()):
    return SimpleNamespace(pack="O", seed=0, truth=copy.deepcopy(g.truth), ptruth=g.ptruth,
                           psnaps={}, config=pack.config, events=list(events),
                           incidents=list(incidents), series={}, scenario_dt=900.0,
                           gen_stats={}, aborted=None)


def test_failed_claims_keep_their_pattern_id(org_run):
    pack, g, pt = org_run
    ipc = M.ip_classes_of(pack.config)
    snap = M.truth_as_statements(pt, 13)
    sts = snap["systems"]["oa"]["model.pviews"]["statements"]
    base = M.pg1_snapshot(copy.deepcopy(snap), pt, ipc, 13, 0, precision_n=100)
    bogus = copy.deepcopy(next(s for s in sts if s["evidence"]["route"] == "POST /login"))
    bogus["evidence"]["route"] = "GET /admin/export"
    bogus["pattern_id"] = "p:oa:0:999"
    sts.append(bogus)
    r = M.pg1_snapshot(snap, pt, ipc, 13, 0, precision_n=100)
    assert len(r["failed"]) == r["n_precision_all"] - round(r["precision_all"] * r["n_precision_all"])
    assert "p:oa:0:999" in {f[0] for f in r["failed"]}
    assert "p:oa:0:999" not in {f[0] for f in base["failed"]}
    # information only: every other figure is unchanged by the list
    assert r["n_precision_all"] == base["n_precision_all"] + 1


def test_far_lists_the_counted_incidents_with_their_constraints(org_run):
    pack, g, pt = org_run
    tr = {r["scenario_id"]: r for r in g.truth}
    t = tr["A1"]["t_first"]
    evs = [{"kind": "pattern_violation", "system": "oa", "entity": "192.168.3.22", "ts": t,
            "severity": "medium", "incident_id": "I2",
            "extra": {"type": "when", "flags": ["slot_empty"], "pattern_id": "p:oa:0:7"}},
           {"kind": "pattern_violation", "system": "oa", "entity": "192.168.3.22", "ts": t + 900,
            "severity": "low", "incident_id": "I2",
            "extra": {"type": "content", "flags": [], "pattern_id": "p:oa:0:7"}}]
    incs = [{"id": "I1", "system": "finance", "entity": "192.168.1.23", "opened": t, "severity": "high",
             "history": [{"ts": t, "severity": "high"}]},                       # a dirty entity: not counted
            {"id": "I2", "system": "oa", "entity": "192.168.3.22", "opened": t, "severity": "low",
             "kinds": ["pattern_violation"],
             "history": [{"ts": t, "severity": "medium"}, {"ts": t + 900, "severity": "low"}]},
            {"id": "I3", "system": "oa", "entity": "192.168.3.23", "opened": t, "severity": "info",
             "history": [{"ts": t, "severity": "info"}]}]                       # below LOW: not counted
    far = M.pg6_far(_fake_run(pack, g, evs, incs), pt)
    assert far["inc_low"] == 1 and far["inc_medium"] == 1
    assert len(far["incidents"]) == far["inc_low"]
    (i2,) = far["incidents"]
    assert i2["id"] == "I2" and i2["entity"] == "192.168.3.22" and i2["severity"] == "medium"
    assert i2["day"] == pt.day_of_ts(t)
    assert i2["constraints"] == ["content", "slot_empty", "when"]
    assert i2["pattern_ids"] == ["p:oa:0:7"] and i2["kinds"] == ["pattern_violation"]
