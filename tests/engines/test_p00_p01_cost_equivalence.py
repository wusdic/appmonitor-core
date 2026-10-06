"""Round 5 cost changes in P00 / P01 are exact.

P00: the rows of an aggregated record that carry no l7 view of their own
share the record's; it is parsed once per record instead of once per row.
The batch must equal the one built when every row carries its own copy of
the view (parsed per row).
P01: the evt.ctx columns are built straight from the per-name value lists
instead of one dict per row replayed through a BatchBuilder; same columns,
same order, same arrays as EV.cols_from_rows."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from helpers import T0, ctx, make_store, obs

from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import pparse as PP
from app.engines.derived import event_context as P01
from app.engines.raw import event_builder as P00
from app.engines.raw.action_token import ActionTokenEngine

ON = {"progressive": {"enabled": True}}


def _records(own_l7: bool):
    out = []
    for k in range(6):
        body = f"username=u{k}&password=pw{k}x&items[]=a&items[]=b&csrf={'ab' * 20}"
        l7 = {"body": body, "body_len": len(body), "query": f"page={k}&q=x",
              "headers": {"content-type": "application/x-www-form-urlencoded", "user-agent": "UA/1",
                          "cookie": f"JSESSIONID=s{k % 2}"}}
        rows = [{"o": float(i), "up": 900.0 + i, "st": 200} for i in range(7)]
        if k == 3:
            rows[2]["l7"] = {"body": "{\"a\": 1, \"b\": [1, 2]}", "body_type": "json",
                             "headers": {"content-type": "application/json"}}
        if k == 4:
            l7 = {"body": object(), "headers": {"content-type": "text/plain"}}   # unparseable body
        if own_l7:
            for r in rows:
                r.setdefault("l7", copy.deepcopy(l7))
        out.append(obs("portal", f"10.0.0.{k}", T0 + k, peer="10.9.9.9", dst_port=443, http_method="POST",
                       http_host="p.local", http_path=f"/form/{k}", http_status=200, bytes_up=1000,
                       extra={"count": 28, "ev_sample": rows, "l7": l7}))
    return out


def _batch(observations, monkeypatch=None):
    st = make_store()
    c = ctx(st, T0 + 60, window_s=60.0,
            config=dict(ON, progressive={"enabled": True, "session_cookies": ["JSESSIONID"]}))
    ActionTokenEngine().safe_run(c, observations)
    eng = P00.EventBuilderEngine()
    eng.safe_run(c, observations)
    return st.batch_at("portal", EV.EVT_BATCH, T0 + 60), eng.last_stats


def _same(a, b):
    assert a.n == b.n
    assert list(a.cols) == list(b.cols)
    for nm in a.cols:
        ca, cb = a.cols[nm], b.cols[nm]
        assert ca.rows.tolist() == cb.rows.tolist(), nm
        assert ca.vals.dtype == cb.vals.dtype, nm
        assert [repr(v) for v in ca.vals.tolist()] == [repr(v) for v in cb.vals.tolist()], nm
    for f in ("ts", "w", "pi", "flags", "learn"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
    assert a.ips == b.ips and a.meta == b.meta


def test_record_l7_parsed_once_and_batch_unchanged(monkeypatch):
    calls = []
    real = PP.parse_l7

    def counting(l7, *a, **kw):
        calls.append(id(l7))
        return real(l7, *a, **kw)
    monkeypatch.setattr(PP, "parse_l7", counting)
    shared, st_shared = _batch(_records(own_l7=False))
    n_shared = len(calls)
    calls.clear()
    own, st_own = _batch(_records(own_l7=True))
    n_own = len(calls)
    _same(shared, own)
    assert st_shared == st_own
    assert st_shared["parse_errors"] == 7                 # record 4's rows, every row counted
    assert n_own == 42 and n_shared == 6 + 1              # one per record (+ the row with its own view)


@pytest.mark.parametrize("seed", range(5))
def test_ctx_columns_equal_rows_replayed_through_a_builder(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 60))
    names = ["ctx.tod_min", "ctx.dow", "ctx.daytype", "ctx.when", "ctx.sid", "ctx.sess_pos",
             "ctx.prev_route", "ctx.think_s", "ctx.flag"]
    pools = [[1.5, 2.0, 600.25], [0, 3, 6], ["workday", "nonworkday"], [("wd", 540), ("nwd", 3)],
             ["a1b2", "c3d4"], [0, 1, 2], ["GET /x", EV.ABSENT, "POST /y"], [0.0, 12.5, EV.ABSENT],
             [True, False, None, EV.ABSENT]]
    cols = {}
    for nm, pool in zip(names, pools):
        absent_p = rng.random() * 0.7
        cols[nm] = [EV.ABSENT if rng.random() < absent_p else pool[int(rng.integers(0, len(pool)))]
                    for _ in range(n)]
    got = P01._ctx_cols(cols, n)
    rows = [{k: cols[k][i] for k in cols if cols[k][i] is not EV.ABSENT} for i in range(n)]
    ref = EV.cols_from_rows(n, rows)
    assert list(got) == list(ref)
    for nm in ref:
        assert got[nm].rows.dtype == ref[nm].rows.dtype and got[nm].rows.tolist() == ref[nm].rows.tolist()
        assert got[nm].vals.dtype == ref[nm].vals.dtype
        assert [repr(v) for v in got[nm].vals.tolist()] == [repr(v) for v in ref[nm].vals.tolist()]
