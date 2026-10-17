"""P04 / P05 on a recorded pack O event stream (round 5, cost and determinism).

tests/data/ptree_stream_O.json.gz: P00's learned rows of pack O's `oa`
(seed 0, days 1-5) with P01's context columns, as recorded from the pipeline
(the /health monitor's rows kept for one tick in six, `ctx.sid` dropped: a
fixture of ~2 200 rows). The stream is replayed through P02 + P05 + P04
(tests/engines/ptree_sim.Sim) for 4 days: the route partition, learning
leaves with split statistics, value slots, held-out records and P05's hourly
evaluations all run on real attribute mixes.

  * the numba kernels (lib/pnumba, optional) and the numpy path learn the
    same trees and selections, bit for bit;
  * the learned models are the recorded ones (GOLDEN, recorded on the
    round-5 code): a later change that alters what is learned from this
    stream must say so here. The round-5 cost rewrites were checked against
    the pre-round-5 code on this stream and on 3 days of the full pack O
    pipeline (every store model identical, apart from the intended changes:
    the reference statement's serial number instead of id() and the
    hash-independent orders);
  * two processes with different PYTHONHASHSEED learn the same models (no
    state depends on the salted iteration order of str / tuple sets).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Dict

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "..", "data", "ptree_stream_O.json.gz")
DAYS = 4.0

# canonical digests of oa's model.ptree / model.attrsel after DAYS days (numba on or off).
# Re-recorded by the round-5 evaluator for one deliberate change: the structural
# drift's pre-alarm loss level restarts with the ADWINs when the coder's encoding
# changes (pattern_tree, test_p04_round5_eval.py); without that restart the
# digest is the tree owner's 3b08ee6b3bc21a29c7ea25afcb0378a43bf95652.
# Round 6 (tree owner): the structural loss is coded per day type and its
# levels / an open alarm restart with the coder, the daily-mean Page-Hinkley
# history is kept per summary scale, unrecorded days are classed from the
# calendar (test_p04_round6.py): model.ptree re-recorded (round 5:
# 86afb654e21302a504e89c9fffadbbe0851c17fc); model.attrsel unchanged.
# Round 6, second change (tree owner): a held-out test judges 'when' on checks
# spanning a day (pnode.HoldRecord trailing-day bins, tests/lib/test_pnode_round6.py):
# model.ptree re-recorded (first round-6 digest 7e48735c1965a0f0a0578c57b7cc65ffd03b1be1)
# and P05's redundancy is judged on the rows where the dependent attribute is
# present too (tests/lib/test_pselect_round6.py): hdr.content-type is no longer
# 'redundant' with body.fmt on this stream (and body.kv.dept / period / items[]
# become split candidates): model.attrsel re-recorded (4b8e9e79e81bdbd30e5fba08d9c2d8c66f15ddee).
# The bounded P05 pools / value groups and P04's pair index do not alter it.
# Round 6, evaluator: P05 compares a text attribute with few values on its
# values, not its shape (tests/lib/test_pselect_round6.py::
# test_a_text_field_is_compared_on_its_values_not_its_shape): on this stream
# hdr.content-type is 'redundant' with body.fmt again (its value is a function
# of the body format; at shape level it was not), and body.kv.dept / period /
# items[] / items[].n are dropped instead of split candidates. Both digests
# re-recorded (before: model.ptree 94de57ce3c306a316cb140b7c6b9d5a6f37c2ace,
# model.attrsel eaf0805815f991e9c5aa07624c0f956b0ce663bd).
# Round 6, evaluator EV6-7: P05's probe interns want sets as bit masks and
# schemas as key tuples (pselect.StratifiedProbe; test_pselect_round6.py::
# test_probe_interning_stays_compact_when_the_wanted_slice_rotates). The model
# is unchanged in content - the same rows, want sets, columns and every other
# field, checked representation-free on this stream under PYTHONHASHSEED 0 and
# 1 - but its canonical form is not: model.attrsel re-recorded (before:
# 4b8e9e79e81bdbd30e5fba08d9c2d8c66f15ddee); model.ptree unchanged.
# Round 7 (tree owner), re-recorded for these deliberate changes (each with its
# own test): P04's held-out record keeps the failed keys per test ('fl' / 'ti',
# M45 forgiveness), the '~when' facet (lite group records too) and clustered
# batch checks as flat arrays; P05 judges redundancy on non-modal values too
# and marks exact / numeric copies beyond the pairwise window
# ('redundant_exact', empty on this stream); DecayedSpaceSaving shares its
# half-life arrays (representation only). With the representation-only parts
# undone, the verdict state equals round 6's on this stream.
GOLDEN = {"model.ptree": "43cbe63182264bd3d863603d25e3fef1303b2249",
          "model.attrsel": "ccf40899e2dff07c5b4938f2f5bf91418dc88c4e"}


def _dec(v: Any) -> Any:
    from app.engines.behavior.lib.phier import Shaped
    if isinstance(v, dict):
        if "S" in v:
            return Shaped(v["S"])
        if "F" in v:
            return float(v["F"])
        if "fs" in v:
            return frozenset(_dec(x) for x in v["fs"])
        if "t" in v:
            return tuple(_dec(x) for x in v["t"])
        if "l" in v:
            return [_dec(x) for x in v["l"]]
        return v.get("r")
    return v


def canon(obj: Any, h: Any, memo: Dict[int, int]) -> None:
    """Order-sensitive canonical feed of a model object graph (floats exact;
    set elements order-free; attributes named _c_* are caches, not state)."""
    t = type(obj)
    if obj is None or isinstance(obj, (bool, int, str, bytes)):
        h.update(f"{t.__name__}:{obj!r};".encode())
        return
    if isinstance(obj, float):
        h.update(f"f:{obj.hex() if obj == obj else 'nan'};".encode())
        return
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        h.update(f"np{t.__name__}:{obj!r};".encode())
        return
    if isinstance(obj, np.ndarray):
        h.update(f"nd:{obj.dtype}:{obj.shape};".encode())
        if obj.dtype == object:
            for x in obj.ravel().tolist():
                canon(x, h, memo)
        else:
            h.update(np.ascontiguousarray(obj).tobytes())
        return
    if callable(obj) and not hasattr(obj, "__dict__") and not hasattr(obj, "__slots__"):
        h.update(f"fn:{t.__name__};".encode())
        return
    if id(obj) in memo:
        h.update(f"ref:{memo[id(obj)]};".encode())
        return
    memo[id(obj)] = len(memo)
    if isinstance(obj, dict):
        h.update(f"{t.__name__}{{{len(obj)};".encode())
        for k, v in obj.items():
            canon(k, h, memo)
            canon(v, h, memo)
        return
    if isinstance(obj, (list, tuple)) or t.__name__ == "deque":
        h.update(f"{t.__name__}[{len(obj)};".encode())
        for v in obj:
            canon(v, h, memo)
        return
    if isinstance(obj, (set, frozenset)):
        subs = []
        for v in obj:
            hh = hashlib.sha1()
            canon(v, hh, {})
            subs.append(hh.hexdigest())
        h.update(f"{t.__name__}<{''.join(sorted(subs))}>".encode())
        return
    h.update(f"obj:{t.__module__}.{t.__name__};".encode())
    names = sorted(vars(obj)) if hasattr(obj, "__dict__") else []
    for c in t.__mro__:
        for sl in getattr(c, "__slots__", ()) or ():
            if sl not in names and sl not in ("__dict__", "__weakref__"):
                names.append(sl)
    for nm in names:
        if nm.startswith("_c_"):
            continue
        try:
            v = getattr(obj, nm)
        except AttributeError:
            continue
        h.update(f".{nm}=".encode())
        canon(v, h, memo)


def replay(days: float = DAYS, numba_on: bool = True) -> Dict[str, str]:
    """Replay the recorded stream; canonical digests of oa's P04 / P05 models."""
    sys.path.insert(0, HERE)
    import ptree_sim as SIM
    from app.engines.behavior.lib import pnumba as PNB
    with gzip.open(FIXTURE, "rt") as f:
        d = json.load(f)
    t0 = float(d["meta"]["start"])
    old = PNB.set_enabled(numba_on)
    try:
        sim = SIM.Sim(dt=900.0, p05=True, t0=t0,
                      config={"progressive": {"enabled": True}, "tz": "Asia/Shanghai"})
        evs = []
        for ts, s, ip, w, pi, a in d["rows"]:
            if ts > t0 + days * 86400.0:
                continue
            attrs = {k: _dec(v) for k, v in a.items()}
            attrs["__w"] = w
            attrs["__pi"] = pi
            evs.append((ts, s, ip, attrs))
        sim.add(evs)
        sim.run_until(t0 + days * 86400.0 + 7200.0)
    finally:
        PNB.set_enabled(old)
    out = {}
    for nm in ("model.ptree", "model.attrsel"):
        h = hashlib.sha1()
        canon(sim.st.get_model("oa", "__system__", nm), h, {})
        out[nm] = h.hexdigest()
    tr = sim.tree("oa")
    out["n_learning"] = sum(1 for nd in tr.nodes.values() if nd.split_stats is not None)
    out["n_nodes"] = len(tr.nodes)
    return out


@pytest.fixture(scope="module")
def numba_run() -> Dict[str, str]:
    return replay(numba_on=True)


def test_the_stream_exercises_learning(numba_run):
    assert numba_run["n_nodes"] >= 8
    assert numba_run["n_learning"] >= 5                    # split statistics were updated


def test_numba_and_numpy_paths_learn_the_same_models(numba_run):
    from app.engines.behavior.lib import pnumba as PNB
    if not PNB.HAVE_NUMBA:
        pytest.skip("numba not installed: the numpy path is the only path")
    assert replay(numba_on=False) == numba_run


def test_models_are_the_recorded_ones(numba_run):
    for nm, want in GOLDEN.items():
        assert want is not None and numba_run[nm] == want, (nm, numba_run[nm])


def test_two_hash_seeds_learn_the_same_models(numba_run):
    """Different PYTHONHASHSEED (salted str / tuple hashes): the same models.
    Before round 5 the set summaries counted a frozenset's elements in hash
    order (which 64 elements, and the Space-Saving eviction order), the prune /
    merge savings summed target pairs in set order, a collapse rebuilt the
    node's targets in set order and P04 kept id() of the reference statement."""
    code = ("import json, sys; sys.path.insert(0, %r); sys.path.insert(0, %r); sys.path.insert(0, %r); "
            "import test_p04_p05_stream_equivalence as T; print(json.dumps(T.replay()))"
            % (HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "..", "backend")))
    outs = []
    for seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                           timeout=600, cwd=HERE)
        assert r.returncode == 0, r.stderr[-2000:]
        outs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    assert outs[0] == outs[1]
    assert outs[0] == numba_run
