"""EventBatch.get (round 3: row-aligned Python lists per column) returns
exactly what the round-2 position-array lookup returned."""
import numpy as np

from app.engines.behavior.lib import pevent as EV


def _ref_get(b, name, row, default=EV.ABSENT):
    """Round-2 EventBatch.get, verbatim."""
    pos = b._position(name)
    if pos is None:
        return default
    j = pos[row]
    if j < 0:
        return default
    v = b.cols[name].vals[j]
    return float(v) if isinstance(v, np.floating) else v


def test_get_matches_the_position_lookup():
    r = np.random.default_rng(3)
    bb = EV.BatchBuilder("s", EV.KIND_TXN)
    for i in range(400):
        a = {}
        if r.random() < 0.7:
            a["num"] = float(r.normal())
        if r.random() < 0.5:
            a["int"] = int(r.integers(0, 9))
        if r.random() < 0.6:
            a["txt"] = str(r.integers(0, 50))
        if r.random() < 0.3:
            a["set"] = frozenset({"a", str(i % 3)})
        if r.random() < 0.4:                       # mixed column: str and numpy float
            a["mix"] = np.float64(i / 7.0) if i % 2 else f"v{i}"
        bb.add(float(i), f"10.0.0.{i % 17}", a)
    b = bb.build(0.0, 900.0)
    for name in ("num", "int", "txt", "set", "mix", "missing"):
        for row in list(range(b.n)) + [-1, -2, np.int64(5), np.int32(7)]:
            for dflt in (EV.ABSENT, None, "x"):
                got, want = b.get(name, row, dflt), _ref_get(b, name, row, dflt)
                assert type(got) is type(want) and (got == want or (got != got and want != want)), \
                    (name, row, got, want)
    # an aligned batch and a selection read their own columns
    sub = b.select(range(0, b.n, 3))
    for row in range(sub.n):
        assert sub.get("num", row) == _ref_get(sub, "num", row)
