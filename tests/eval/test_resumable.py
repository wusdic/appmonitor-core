"""eval/resumable.py: a run interrupted at checkpoints and resumed (in a fresh
load of the pickled state each time) gives the same result as runner.run_pack;
only wall-time measurements differ (they differ between two plain runs too)."""
import io
import json
import re

import numpy as np

from app.eval import packs as P
from app.eval import pmetrics as M
from app.eval import runner as R
from app.eval.resumable import run_pack_resumable
from app.pipeline import orggen as G

# wall-time measurements the engines publish in their models
_TIMING = re.compile(r"(^|/)(ms|ms_[a-z_]+|us_[a-z_]+|[a-z0-9_]+_ms|[a-z0-9_]*cpu_share|"
                     r"[a-z0-9_]*_us|wall_s|ts_wall)$|/(cpu|timing|gain/ms|gain/us_per_event)(/|$)")


def _flat(x, pre=""):
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(_flat(v, f"{pre}/{k}"))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            out.update(_flat(v, f"{pre}[{i}]"))
    else:
        out[pre] = repr(x)
    return out


def test_resumed_run_equals_run_pack(tmp_path):
    pack = P._org_pack("O", G.build_org("O", n_days=2, portal_n=20), "resumable test",
                       phases=[(104, 900.0)])          # one day + 2 h: one day-end checkpoint
    ck = str(tmp_path / "O_0.ckpt")
    n_calls = 0
    res = None
    while res is None:
        n_calls += 1
        res = run_pack_resumable(pack, 0, ck, segment_s=0.0, stop_after_s=0.0)
        assert n_calls < 5
    assert n_calls == 2 and res.segments == 1
    ref = R.run_pack(pack, 0, record_series=False)
    fa = _flat(json.loads(json.dumps(ref.psnaps, default=str)))
    fb = _flat(json.loads(json.dumps(res.psnaps, default=str)))
    assert set(fa) == set(fb)
    # P12's arm costs and P15's budget weights are wall-clock measurements:
    # they differ between two plain runs as well; the arms P12 chose do not
    wall = lambda k: _TIMING.search(k) or "/model.sysprof/" in k and "/chosen/" not in k \
        or "/model.budget/" in k
    diff = [k for k in fa if fa[k] != fb[k] and not wall(k)]
    assert not diff, diff[:20]
    for d in ref.psnaps:
        for sysid, m in ref.psnaps[d]["systems"].items():
            assert (m.get("model.sysprof") or {}).get("chosen") == \
                (res.psnaps[d]["systems"][sysid].get("model.sysprof") or {}).get("chosen")
    sa, sb = M.score_prun(ref), M.score_prun(res)
    assert json.dumps(sa, sort_keys=True, default=str) == json.dumps(sb, sort_keys=True, default=str)
    assert [e["id"] for e in ref.events] == [e["id"] for e in res.events]
    assert len(ref.incidents) == len(res.incidents) and not res.exceptions
    assert np.array_equal(ref.tick_ts, res.tick_ts)
    # the per-tick taps of PG4 (generator events, P00 rows)
    assert len(res.tap["events"]) == 104 and sum(res.tap["events"]) == res.gen_stats["events"]


def test_checkpoint_keeps_identity_sentinels():
    """Engines test stored values with 'is EV.ABSENT' (and pselect.MISSING,
    likelihood._UNSET ...): pickled by value they came back as equal but
    distinct objects, and P07 counted stored absences as a key after a resume."""
    import threading
    import weakref
    from app.engines.behavior.lib import pevent as EV
    from app.engines.behavior.lib import pselect as PSEL
    from app.eval import resumable as RS

    class Holder:
        pass
    h = Holder()
    h.vals = [EV.ABSENT, PSEL.MISSING, "⊥ but not the sentinel"[:1]]
    h.lock = threading.RLock()
    h.ref = weakref.ref(h)
    h.wkd = weakref.WeakKeyDictionary({h: 1})
    h.fn = lambda x: x + 1
    buf = io.BytesIO()
    RS.ckpt_pickler(buf).dump(h)
    buf.seek(0)
    g = RS.ckpt_load(buf)
    assert g.vals[0] is EV.ABSENT and g.vals[1] is PSEL.MISSING
    assert g.ref() is g and g.wkd[g] == 1 and g.fn(1) == 2
    with g.lock:
        pass
