"""Round 7 (evaluation-harness owner, input cost): P00's work per event must
not grow by more than the fields an event carries (PG4 attribute axis: CPU
per event x3.1 for P00 from 40 to 340 attributes, x4.2 per input record).

* P00 flattens adapter / WAF meta fields through `event_builder.flatten_meta`:
  the same names, values, order and return value as `pevent.flatten`, with
  the leaf names memoised per (prefix, key) for flat scalar dicts.
* The value policy's `apply_all` skips values it passes unchanged under
  every mode (int / float / bool / None): same result.
Exactness is checked against the reference implementations; cost by
operation counts, not wall time. (P02's growth is set by how many attributes
P05 keeps - every kept attribute is observed in full - see eval.md round 7.)"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import pparse as PP
from app.engines.raw import event_builder as P00


class _Sub(dict):
    pass


def _cases():
    rng = np.random.default_rng(5)
    out = []
    for k in range(40):
        d = {}
        for j in range(int(rng.integers(0, 90))):            # more than 64 leaves too
            key = f"F{j:03d}" if j % 5 else f"a b.{j}[2]/X"     # names normalised / templated
            x = int(rng.integers(0, 7))
            d[key] = [None, True, 3, 2.5, "Val", "", -0.0][x]
        out.append(d)
    out += [{"x": {"y": 1}}, {"a": 1, "b": [1, 2], "c": [{"k": "v"}]}, {"n": np.float64(2.0)},
            {1: "int key", "1": "str key"}, _Sub(a=1), {"A": 1, "a": 2}, {}, {"s": "x" * 300}]
    return out


@pytest.mark.parametrize("max_leaves", [64, 3, 0])
def test_flatten_meta_equals_pevent_flatten(max_leaves):
    for case in _cases():
        for prefix in ("meta", "x"):
            want: dict = {"pre": 0}
            got: dict = {"pre": 0}
            n_want = EV.flatten(prefix, copy.deepcopy(case), want, max_leaves=max_leaves)
            n_got = P00.flatten_meta(prefix, copy.deepcopy(case), got, max_leaves=max_leaves)
            assert n_got == n_want
            assert list(got.items()) == list(want.items())
            assert [type(v) for v in got.values()] == [type(v) for v in want.values()]


def test_flatten_meta_names_are_computed_once_per_key(monkeypatch):
    calls = []
    real = EV.attr_name
    monkeypatch.setattr(EV, "attr_name", lambda *a: calls.append(a) or real(*a))
    P00._LEAF_NAMES.clear()
    meta = {f"f{j:03d}": float(j) for j in range(64)}
    for _ in range(50):                                      # 50 events with the same fields
        P00.flatten_meta("meta", dict(meta), {}, max_leaves=64)
    assert len(calls) == 64                                   # not 50 x 64


def _apply_all_ref(vp, attrs, modes):
    """pparse.ValuePolicy.apply_all before round 7 (reference)."""
    extra = {}
    for nm in list(attrs.keys()):
        if not nm.startswith(PP.PAYLOAD_NS):
            continue
        v = attrs[nm]
        nv, mode = vp.apply(nm, v)
        if mode != "clear":
            attrs[nm] = nv
            modes[nm] = mode
            if isinstance(v, str) and mode == "shape" and not nm.endswith(".len"):
                extra[nm + ".len"] = len(v)
    attrs.update(extra)
    return attrs


def test_value_policy_apply_all_unchanged():
    cfg = {"rules": [["hdr.authorization", "hmac"], ["meta.secret*", "shape"]],
           "secret_globs": ["*password*"]}
    rows = []
    for k in range(30):
        rows.append({"meta.f1": float(k), "meta.f2": k, "meta.flag": bool(k % 2), "meta.none": None,
                     "meta.secret_x": f"s{k}", "body.kv.password": f"pw{k}", "hdr.authorization": "Bearer x",
                     "body.keys": frozenset({"a", "password"}), "net.bytes_up": 3.0,
                     "q.kv.tok": "a3f9c2e81b7d4c6e9f0a1b2c3d4e5f60", "meta.long": "y" * 100})
    for r in rows:
        a, b = dict(r), dict(r)
        ma, mb = {}, {}
        PP.ValuePolicy(cfg).apply_all(a, ma)
        _apply_all_ref(PP.ValuePolicy(cfg), b, mb)
        assert list(a.items()) == list(b.items()) and ma == mb
