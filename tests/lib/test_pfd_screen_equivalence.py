"""P08 cost (round 5): pfd.screen on factorised columns must return exactly
what the per-pair dict form returned.

The reference below is round 4's screen (every (X, Y, stratum) test on
dicts of Python values). tests/data/pfd_screen_calls_packO.pkl.gz holds 9
recorded screen calls of pack O (O-scale-500-0, seed 0, days 1-2: finance,
crm, oa, portal; 21 - 3524 probe rows) with their inputs (rows, weights,
candidates, strata, the generalisation table and the payload cardinalities).
"""
from __future__ import annotations

import gzip
import math
import os
import pickle
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytest

from app.engines.behavior.lib import pfd as FD
from app.engines.behavior.lib.phier import Shaped

CALLS = os.path.join(os.path.dirname(__file__), "..", "data", "pfd_screen_calls_packO.pkl.gz")


def screen_ref(rows: Sequence[Mapping[str, Any]], w: np.ndarray, x_cands: Sequence[Tuple[str, int]],
               y_cands: Sequence[str], gen: Callable[[str, int, Any], Any],
               q_pairs: int = FD.Q_PAIRS, absent: Any = None,
               strata: Optional[Sequence[str]] = None,
               card: Optional[Callable[[str], float]] = None) -> List[Dict[str, Any]]:
    """Round 4's pfd.screen."""
    out = []
    xcols: Dict[Tuple[str, int], List[Any]] = {}
    for (xa, xl) in x_cands:
        col = []
        memo: Dict[Any, Any] = {}
        for r in rows:
            xv = r.get(xa, absent)
            if xv is absent or xv is None:
                col.append(None)
                continue
            try:
                g = memo.get(xv, memo)
                if g is memo:
                    g = memo[xv] = gen(xa, xl, xv)
            except TypeError:
                g = gen(xa, xl, xv)
            col.append(None if g is absent else g)
        xcols[(xa, xl)] = col
    wl = [float(x) for x in w]
    for yname in y_cands:
        idx = []
        yv_all = {}
        for i, r in enumerate(rows):
            v = r.get(yname, absent)
            if v is not absent and v is not None and not isinstance(v, Shaped) \
                    and isinstance(v, (str, int, float, bool)):
                idx.append(i)
                yv_all[i] = v
        if len(idx) < 2 * FD.SCREEN_MIN_ROWS:
            continue
        for (xa, xl) in x_cands:
            col = xcols[(xa, xl)]
            xs, ys, ww = [], [], []
            for i in idx:
                g = col[i]
                if g is None:
                    continue
                xs.append(g)
                ys.append(yv_all[i])
                ww.append(wl[i])
            if len(xs) < 2 * FD.SCREEN_MIN_ROWS:
                continue
            xn = FD.x_name(xa, xl)
            groups: Dict[Optional[str], List[int]] = {None: list(range(len(xs)))}
            if strata is not None:
                for j, i in enumerate(i for i in idx if col[i] is not None):
                    groups.setdefault(str(strata[i]), []).append(j)
            for gk, sel in groups.items():
                if len(sel) < 2 * FD.SCREEN_MIN_ROWS or (gk is not None and len(groups) == 2):
                    continue
                gx = [xs[j] for j in sel]
                gy = [ys[j] for j in sel]
                gw = [ww[j] for j in sel]
                id_like = card is None or float(card(yname)) >= FD.SET_MIN_CARD
                st = FD.screen_pair(gx, gy, gw)
                if st is not None and (st["fd"] or (st["set"] and id_like)):
                    out.append({"x": xn, "y": yname, "dir": "fwd", "stats": st, "strata": [gk]})
                rs = FD.screen_pair(gy, gx, gw)
                if rs is not None and (rs["fd"] or (rs["set"] and id_like)):
                    out.append({"x": yname, "y": xn, "dir": "rev", "stats": rs, "strata": [gk]})
    out.sort(key=lambda d: (-(1.0 - d["stats"]["g3"]) * max(d["stats"].get("lambda", 0.0), 0.0)
                            * min(d["stats"]["hy"], 4.0), d["x"], d["y"]))
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    res = []
    for d in out:
        k = (d["x"], d["y"])
        if k in seen:
            seen[k]["strata"] = sorted(set(seen[k]["strata"]) | set(d["strata"]), key=str)
            continue
        if len(res) >= q_pairs:
            continue
        d = dict(d, strata=list(d["strata"]))
        seen[k] = d
        res.append(d)
    for d in res:
        d["strata"] = [x for x in d["strata"] if x is not None]
    return res


def _exact(o: Any) -> Any:
    if isinstance(o, float):
        return ("f", "nan" if o != o else o.hex())
    if isinstance(o, dict):
        return ("d", [(k, _exact(v)) for k, v in o.items()])
    if isinstance(o, (list, tuple)):
        return (type(o).__name__, [_exact(v) for v in o])
    return (type(o).__name__, o)


@pytest.fixture(scope="module")
def calls():
    with gzip.open(CALLS, "rb") as f:
        return pickle.load(f)


def _run(fn, c, **kw):
    g = c["gtab"]
    return fn(c["rows"], c["w"], c["x_cands"], c["y_cands"], lambda a, l, v: g[(a, l, v)],
              q_pairs=c["q_pairs"], absent=c["absent"], strata=c["strata"],
              card=lambda y: c["ctab"][y], **kw)


@pytest.mark.parametrize("dense", [True, False])
def test_screen_is_identical_on_recorded_pack_o_calls(calls, dense, monkeypatch):
    if not dense:
        monkeypatch.setattr(FD, "DENSE_CELLS", 0)
    found = 0
    for c in calls:
        new, ref = _run(FD.screen, c), _run(screen_ref, c)
        assert _exact(new) == _exact(ref)
        found += len(ref)
        # and without strata (the pooled tests only)
        c2 = dict(c, strata=None)
        assert _exact(_run(FD.screen, c2)) == _exact(_run(screen_ref, c2))
    assert found >= 20                     # the fixture exercises accepted pairs


@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize("seed", range(6))
def test_screen_pair_codes_match_dict_form_on_random_data(seed, dense, monkeypatch):
    """Mixed value types that share dict keys (1, 1.0, True), shared:
    sources, ties and duplicate rows: the code form gives the dict form's
    statistics float for float, both directions, any subset size."""
    if not dense:
        monkeypatch.setattr(FD, "DENSE_CELLS", 0)          # the sparse (np.unique) table
    rng = np.random.default_rng(seed)
    pool_x = ["10.0.0.%d" % i for i in range(12)] + ["shared:10.0.0.99", 1, 1.0, True]
    pool_y = ["jack", "rose", "mike", 7, 7.0, False, 0, "x" * 3]
    n = int(rng.integers(6, 400))
    xs = [pool_x[int(i)] for i in rng.integers(0, len(pool_x), n)]
    ys = [pool_y[int(i)] for i in rng.integers(0, len(pool_y), n)]
    w = (rng.random(n) * rng.choice([1.0, 0.1, 1e3], n)).tolist()
    xc, xr = FD._factorize(xs)
    yc, yr = FD._factorize(ys)
    xsh = np.asarray([FD.is_shared(v) for v in xr], dtype=bool)
    ysh = np.asarray([FD.is_shared(v) for v in yr], dtype=bool)
    wa = np.asarray(w)
    for sub in (np.arange(n), np.flatnonzero(rng.random(n) < 0.5)):
        if sub.size == 0:
            continue
        gx = [xs[i] for i in sub]
        gy = [ys[i] for i in sub]
        gw = [w[i] for i in sub]
        a = FD._screen_pair_codes(xc[sub], yc[sub], wa[sub], len(xr), len(yr), xsh)
        assert _exact(a) == _exact(FD.screen_pair(gx, gy, gw))
        b = FD._screen_pair_codes(yc[sub], xc[sub], wa[sub], len(yr), len(xr), ysh)
        assert _exact(b) == _exact(FD.screen_pair(gy, gx, gw))
