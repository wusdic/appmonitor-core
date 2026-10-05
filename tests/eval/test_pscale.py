"""PG4 scaling harness and the runner's progressive snapshots, validated with
toy engines (the P engines do not exist yet): a bounded learner must measure a
log-log memory slope ~0 against the number of IPs, a per-IP learner ~1."""
from __future__ import annotations

import zlib

import numpy as np
import pytest

from app.core.engine import Engine, Registry
from app.eval import packs as P
from app.eval import pscale as S
from app.eval.runner import run_pack
from app.models.schema import SYSTEM_ENTITY


class ToyLearner(Engine):
    """Writes model.ptree per system: a fixed 64-slot count array (bounded) or
    one node per client IP (per-IP state, what the requirement forbids)."""

    name = "raw.event"
    layer = "raw"

    def __init__(self, per_ip: bool) -> None:
        super().__init__()
        self.per_ip = per_ip

    def run(self, ctx, observations=None):
        st = ctx.store
        for o in observations or []:
            m = st.get_model(o.system, SYSTEM_ENTITY, "model.ptree")
            if m is None:
                m = {"kinds": {"0": {"nodes": {}}}, "counts": np.zeros(64)}
                st.put_model(o.system, SYSTEM_ENTITY, "model.ptree", m, ts=ctx.now)
            if self.per_ip:
                node = m["kinds"]["0"]["nodes"].setdefault(o.entity, {"ctx": [], "n": np.zeros(8)})
                node["n"][0] += 1
            else:
                m["counts"][zlib.crc32(o.entity.encode()) % 64] += 1
        return len(observations or [])


class ToyViews(Engine):
    name = "behavior.views"
    layer = "behavior"

    def run(self, ctx, observations=None):
        ctx.store.put_model("oa", SYSTEM_ENTITY, "model.pviews", {"statements": [
            {"pattern_id": "p:oa:0:1", "state": "confirmed", "evidence": {"route": "POST /login"},
             "now": ctx.now}]}, ts=ctx.now)
        return 1


def _reg(*engines):
    r = Registry()
    r.add(*engines)
    return lambda **kw: r


@pytest.mark.parametrize("per_ip,lo,hi", [(False, -0.05, 0.1), (True, 0.5, 1.1)])
def test_memory_slope_vs_ips(per_ip, lo, hi):
    pts = [S.run_point(P.pack_o_scale(n, 0, f"O-scale-{n}-0", n_days=1), 0,
                       registry_factory=_reg(ToyLearner(per_ip)))
           for n in (100, 400, 1600)]
    assert all(p["exceptions"] == 0 and p["aborted"] is None for p in pts)
    assert [p["n_ips"] for p in pts] == [100, 400, 1600]
    slope = S._slope(pts, "n_ips", lambda p: p["mem_bytes"])
    assert lo <= slope <= hi, slope
    assert all(p["cpu"]["us_per_event"] and p["cpu"]["us_per_event"] > 0 for p in pts)
    g = S.pg4_summary(pts)
    first = g["details"]["checks"][0]
    assert first["pass"] is (not per_ip)


def test_deep_sizeof_counts_arrays_once():
    a = np.zeros(1000)
    assert S.deep_sizeof({"x": a, "y": a}) < S.deep_sizeof({"x": a, "y": np.zeros(1000)})
    assert S.deep_sizeof([a]) >= 8000


def test_runner_progressive_snapshots_and_truth():
    pack = P.pack_o_scale(50, 0, "O-scale-t", n_days=2)
    pack.full_snapshot_days = [2]
    res = run_pack(pack, 0, registry_factory=_reg(ToyViews()), record_series=False)
    assert res.aborted is None and not res.exceptions
    assert sorted(res.psnaps) == [1, 2]
    assert res.psnaps[1]["full"] is False and res.psnaps[2]["full"] is True
    st = res.psnaps[2]["systems"]["oa"]["model.pviews"]["statements"][0]
    assert st["now"] == pytest.approx(res.psnaps[2]["ts"])
    assert res.ptruth["pattern_truth"] and res.ptruth["who_log"]["oa"]
    assert res.gen_stats["events"] > 1000
    assert all(r["t_start"] < pack.end_epoch for r in res.truth)     # D1..A10 lie beyond day 2
    from app.eval.pmetrics import compute_pgates, score_prun
    sc = score_prun(res, precision_n=20)
    assert sc["core_active"] and sc["n_snapshots"] == 2 and sc["pg1"]["2"]["n_stmt"] == 1
    assert compute_pgates([sc])["PG1"]["name"].startswith("PG1")


def test_pcore_cpu_units_scored_and_learned_events():
    """§12 PG4: scoring p95 per SCORED event, learning p95 per LEARNED event.
    10 ms of P04 per tick over 1 000 generated, 1 000 scored and 100 learned
    events is 100 us per learned event (10 us per generated event before)."""
    t = {"engine_names": ["behavior.conformity", "behavior.pattern_tree"],
         "engine_ms": [[5.0, 10.0]] * 4}
    old = S.pcore_cpu(t, [1000.0] * 4)
    new = S.pcore_cpu(t, [1000.0] * 4, batch_per_tick=[(1000.0, 100.0)] * 4)
    assert old["learning_p95_us"] == pytest.approx(10.0)
    assert new["learning_p95_us"] == pytest.approx(100.0)
    assert new["scoring_p95_us"] == pytest.approx(5.0)
    assert new["learning_us_per_learned_event"] == pytest.approx(100.0)


def test_entry_points_pin_the_hash_seed(tmp_path):
    """Two processes of the same evaluation must iterate str sets in the same
    order (§16.12): pin_hash_seed re-executes an unpinned script with
    PYTHONHASHSEED set, so hash() agrees across processes."""
    import os
    import subprocess
    import sys
    backend = os.path.join(os.path.dirname(__file__), "..", "..", "backend")
    script = tmp_path / "h.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {os.path.abspath(backend)!r})\n"
        "from app.eval.pscale import pin_hash_seed\n"
        "pin_hash_seed()\n"
        "print(hash('progressive-core'), list({'a%d' % i for i in range(20)})[:5])\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    out = [subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True,
                          timeout=120).stdout for _ in range(3)]
    assert out[0] and out[0] == out[1] == out[2]
    # unpinned processes disagree (the salt is random per process)
    plain = tmp_path / "p.py"
    plain.write_text("print(hash('progressive-core'))\n")
    outs = {subprocess.run([sys.executable, str(plain)], env=env, capture_output=True, text=True,
                           timeout=60).stdout for _ in range(4)}
    assert len(outs) > 1
