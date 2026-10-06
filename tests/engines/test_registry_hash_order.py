"""Round 5: P02 / P06 / P07 / P08 state must not depend on Python's per-process
str-hash salt (PYTHONHASHSEED).

docs/lib3/progressive.md §16.12.6: identical pack O runs differed between
processes in P02's set-element sketch (and through it set_keep), P07's key
presences, P08's bound shares / g3 (masses summed over a set of sources) and
P06's violation ledger, unless the evaluation pinned PYTHONHASHSEED. Each
case below runs the same computation in fresh interpreters with different
hash seeds and requires byte-identical output (each fails on the round-4
code)."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
SEEDS = ("0", "1", "2", "3", "77")


def _outputs(code: str) -> set:
    outs = set()
    for seed in SEEDS:
        env = dict(os.environ, PYTHONHASHSEED=seed, OMP_NUM_THREADS="1")
        env["PYTHONPATH"] = BACKEND
        r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env,
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        outs.add(r.stdout)
    return outs


def test_registry_set_elements_do_not_depend_on_hash_seed():
    """P02: a key-set attribute with more distinct keys than the element
    sketch holds (body.keys of a form with 40 fields, 32 taken per row,
    TOP_K = 32 slots): which elements are counted and which slot each holds
    followed the set's iteration order."""
    code = """
    import numpy as np
    from app.engines.behavior.lib import pregistry as RG
    rng = np.random.default_rng(0)
    words = ["k%02d_%s" % (i, "abcdefgh"[i % 8] * (1 + i % 3)) for i in range(48)]
    reg = RG.AttrRegistry("oa")
    t = 1.75e9
    for step in range(30):
        vals = [frozenset(rng.choice(words, size=int(rng.integers(20, 44)), replace=False).tolist())
                for _ in range(12)]
        reg.observe("body.keys", vals, t + 900.0 * step, rng.random(12) + 0.5)
    reg.update_types(t + 1e5)
    reg.refresh_hierarchies(t + 1e5, force=True)
    rec = reg.records["body.keys"]
    print(rec.type, rec.elem.keys(), [r[0].hex() for r in rec.elem._m], sorted(rec.hier.get("set_keep") or ()))
    """
    outs = _outputs(code)
    assert len(outs) == 1, outs


def test_binding_fit_does_not_depend_on_hash_seed():
    """P08: fit_pair summed the judged sources' masses over a set of their
    str keys (bound_share, g3) and classify returned its verdicts in a set's
    order (the 'superseded' / 'pending' lists of a table entry)."""
    code = """
    import numpy as np
    from app.engines.behavior.lib import pfd as FD
    from app.engines.behavior.lib import pnode as PN
    rng = np.random.default_rng(1)
    ps = PN.PairSketch()
    t = 1.75e9
    users = ["u%d" % i for i in range(30)]
    for d in range(12):
        for i in range(30):
            ip = "192.168.%d.%d" % (i % 3, 10 + i)
            for _ in range(int(rng.integers(1, 4))):
                y = users[i] if rng.random() < 0.97 else users[(i + 1) % 30]
                ps.update(ip, y, t + d * 86400.0 + float(rng.random() * 3600), float(rng.random() * 0.3 + 0.1), 1.0)
    now = t + 12 * 86400.0
    excl = {"192.168.0.10": {"u1": "pending", "u0x": "superseded"}}
    rec = FD.fit_pair(ps, now, exclude=excl)
    fd = rec["fd"]
    print(fd["bound_share"].hex(), fd["g3"].hex(), fd["g3_all"].hex(), fd["n"].hex(), rec["gain"].hex())
    hx = {"v": {"mike": [t, t + 3 * 86400, 9, [1, 2, 3]], "mike.w": [t + 5 * 86400, now, 9, [6, 7, 8]],
                "jack": [t + 2 * 86400, t + 9 * 86400, 2, [3]], "rose": [t + 3 * 86400, t + 9 * 86400, 2, [4]],
                "kate": [t + 4 * 86400, t + 9 * 86400, 1, [5]]},
          "recent": ["mike.w"] * 5}
    print(list(FD.classify(hx, ["kate", "rose", "jack", "mike", "mike.w"], now).items()))
    print(list(FD.pair_counts(ps, now, exclude=excl)["192.168.0.10"]["y"].items()))
    """
    outs = _outputs(code)
    assert len(outs) == 1, outs


def test_key_presence_order_does_not_depend_on_hash_seed():
    """P07: fit_set's presence map followed the node sketch's slot order,
    which follows the key sets' iteration order."""
    code = """
    from app.engines.behavior.lib import pgrammar as PG
    from app.engines.behavior.lib import pnode as PN
    ss = PN.SetSummary()
    t = 1.75e9
    keys = ["username", "password", "csrf", "captcha", "viewstate", "otp", "remember", "lang"]
    for i in range(400):
        ks = frozenset(k for j, k in enumerate(keys) if (i + j) % 5 or j < 3)
        ss.update(ks, t + 60.0 * i, 1.0, 1.0)
    rec = PG.fit_set(ss, t + 60.0 * 400)
    print(list(rec["presence"]), rec["required"], rec["optional"])
    """
    outs = _outputs(code)
    assert len(outs) == 1, outs


def test_violation_ledger_order_does_not_depend_on_hash_seed():
    """P06: the ledger's per-row value dicts were built over the set of
    tracked numeric attributes (a set of str)."""
    code = """
    import numpy as np
    from app.core.engine import Context
    from app.core.store import MetricStore
    from app.engines.behavior import content_bounds as CB
    from app.engines.behavior.lib import pevent as EV
    from app.models.schema import SYSTEM_ENTITY
    store = MetricStore()
    now = 1.75e9
    attrs = ["net.bytes_up", "net.bytes_down", "net.dur_ms", "body.len", "net.pkts_up", "net.pkts_down",
             "body.kv.viewstate.len", "net.resp_len"]
    b = EV.BatchBuilder("portal")
    a = EV.BatchBuilder("portal")
    for i in range(6):
        b.add(now - 100 + i, "10.0.0.%d" % i, {nm: float(100 * i + j) for j, nm in enumerate(attrs)})
        a.add(now - 100 + i, "10.0.0.%d" % i, {"leaf": 3.0, "vtype": 1.0 if i % 2 else 0.0})
    bb = b.build(now - 900, now)
    store.add_batch("portal", EV.EVT_BATCH, now, bb)
    store.add_batch("portal", EV.PAT_ASSIGN, now, bb.aligned(a.build(now - 900, now).cols))
    store.put_model("portal", SYSTEM_ENTITY, CB.STATE,
                    {"last": {}, "led": __import__("collections").deque(maxlen=CB.LEDGER_MAX),
                     "attrs": sorted(attrs)})
    ctx = Context(store=store, now=now, window_s=900.0, config={})
    CB.ContentBoundsEngine()._ledger(ctx, "portal", now)
    st = store.get_model("portal", SYSTEM_ENTITY, CB.STATE)
    print([list(v) for _, _, _, v in st["led"]])
    """
    outs = _outputs(code)
    assert len(outs) == 1, outs
