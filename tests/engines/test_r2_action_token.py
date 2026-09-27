"""R2 ActionTokenEngine (engines/raw/action_token.py) and its accessor module
lib/m_template.py.

Spec unit tests (docs/lib3/engines.md, R2):
  (a) 3000 /orders/view/<id> + 2000 random 8-char 404 paths -> <= 60 templates,
      /orders/view/{num} holds >= 95 % of the orders mass;
  (b) 2000 events in one tick -> <= 512 rows, stream_frac = kept/2000, no session
      split at the cut;
  (c) a destination used by 1 of 10 entities -> act.rare_events holds all its events;
  (d) a silent known entity -> act.events = 0 at ctx.now, last_seen unchanged;
  (e) act.objs lists the ids, n = 2000, HLL within 5 % (fixed id set, per the
      helper-review correction).
Plus: aggregated observations, token/stack/dest ids, NaN inputs, pseudo and
probe observations, training mode, cadence 900 -> 60, serialisation, perf.
"""
from __future__ import annotations

import json
import math
import random
import string
import time

import numpy as np
import pytest

from helpers import T0, make_store, obs, run_engine

from app.engines.behavior.lib import m_template as MT
from app.engines.behavior.lib import template as TPL
from app.engines.behavior.lib.stack import stack_id, stack_token
from app.engines.derived import fresh
from app.engines.raw.action_token import ActionTokenEngine
from app.models.schema import AcquisitionMethod

S = "erp"
HOST = "erp.corp"


def _http(e, ts, path, status=200, method="GET", host=HOST, **kw):
    return obs(S, e, ts, app_proto="http", http_method=method, http_host=host,
               http_path=path, http_status=status, peer="10.9.0.10", dst_port=80,
               bytes_up=kw.pop("bytes_up", 300), bytes_down=kw.pop("bytes_down", 5000), **kw)


def _tls(e, ts, sni, up=800, down=4000, **kw):
    return obs(S, e, ts, app_proto="tls", tls_sni=sni, dst_port=443, peer="203.0.113.7",
               bytes_up=up, bytes_down=down, **kw)


def _raw(st, e, name, ts, s=S):
    m = st.latest_raw_at(s, e, name, ts)
    return None if m is None else m.value


def _run(eng, st, now, observations, dt=900.0, training=False):
    return run_engine(eng, st, now, training=training, dt=dt, observations=observations)


# ------------------------------------------------------------------ spec (a)
def test_a_orders_flood_templates():
    rng = random.Random(7)
    alphabet = string.ascii_lowercase + string.digits
    reqs = [("o", f"/orders/view/{rng.randrange(1, 10 ** 7)}", 200) for _ in range(3000)]
    reqs += [("j", "/" + "".join(rng.choice(alphabet) for _ in range(8)), 404)
             for _ in range(2000)]
    rng.shuffle(reqs)
    st, eng = make_store(), ActionTokenEngine()
    seen, orders_mass = set(), 0
    for k in range(10):                               # 10 ticks of 500 requests
        now = T0 + k * 900
        batch = [_http("10.0.0.1", now - 800 + i, p, st_)
                 for i, (_, p, st_) in enumerate(reqs[k * 500:(k + 1) * 500])]
        _run(eng, st, now, batch)
        toks = _raw(st, "10.0.0.1", "act.tokens", now)
        assert MT.OTHER not in toks
        assert sum(toks.values()) == 500
        assert _raw(st, "10.0.0.1", "act.distinct_templates", now) <= 60
        seen.update(toks)
        orders_mass += toks.get(f"GET {HOST} /orders/view/{{num}}|2xx", 0)
    assert len(seen) <= 60, sorted(seen)
    assert MT.vocab_size(st, S) <= 60
    assert orders_mass / 3000 >= 0.95


# ------------------------------------------------------------------ spec (b)
def test_b_stream_cap_keeps_whole_sessions():
    rng = random.Random(3)
    st, eng = make_store(), ActionTokenEngine()
    now = T0 + 3600
    batch, sess_of, t, sid = [], {}, T0, 0
    while len(batch) < 2000:
        size = min(rng.randint(5, 40), 2000 - len(batch))
        for _ in range(size):
            t += rng.uniform(0.1, 3.0)
            batch.append(_http("10.0.0.1", t, f"/p/{rng.randrange(5)}"))
            sess_of[t] = sid
        sid += 1
        t += rng.uniform(31.0, 90.0)                  # > 30 s gap = new session
    sizes = np.bincount(list(sess_of.values()))
    rng.shuffle(batch)                                 # arrival order must not matter
    _run(eng, st, now, batch, dt=3600)
    rows = _raw(st, "10.0.0.1", "act.stream", now)
    assert rows.dtype == MT.STREAM_DTYPE and not rows.flags.writeable
    assert 0 < len(rows) <= 512
    assert len(rows) >= 512 - sizes.max()             # greedy packing fills the cap
    assert np.all(np.diff(rows["ts"]) >= 0)
    assert _raw(st, "10.0.0.1", "act.stream_frac", now) == pytest.approx(len(rows) / 2000)
    assert _raw(st, "10.0.0.1", "act.events", now) == 2000
    kept = np.bincount([sess_of[float(x)] for x in rows["ts"]], minlength=len(sizes))
    assert np.all((kept == 0) | (kept == sizes)), "a session was split at the cut"
    # deterministic: same input, same selection
    st2 = make_store()
    _run(ActionTokenEngine(), st2, now, batch, dt=3600)
    assert np.array_equal(_raw(st2, "10.0.0.1", "act.stream", now), rows)


def test_b_oversize_session_keeps_one_contiguous_chunk():
    st, eng = make_store(), ActionTokenEngine()
    now = T0 + 3600
    batch = [_http("10.0.0.1", T0 + i * 1.0, "/api/poll") for i in range(2000)]
    _run(eng, st, now, batch, dt=3600)
    rows = _raw(st, "10.0.0.1", "act.stream", now)
    assert len(rows) == 512
    assert np.allclose(np.diff(rows["ts"]), 1.0)     # contiguous: every gap is real
    assert _raw(st, "10.0.0.1", "act.stream_frac", now) == pytest.approx(512 / 2000)


# ------------------------------------------------------------------ spec (c)
def test_c_rare_destination_events_all_listed():
    st, eng = make_store(), ActionTokenEngine()
    now = T0 + 900
    ents = [f"10.0.1.{i}" for i in range(10)]
    batch = []
    for i, e in enumerate(ents):
        batch += [_http(e, T0 + 10 * i + j, "/home") for j in range(5)]
    rare_ts = [T0 + 100.25 + 61.5 * k for k in range(7)]    # sub-second times
    batch += [_tls(ents[0], t, "c2.rare-dest.example.com", up=100 + k, down=50)
              for k, t in enumerate(rare_ts)]
    _run(eng, st, now, batch)
    did = MT.dest_id_of("example.com")
    assert MT.dest_name(st, S, did) == "example.com"
    ev = _raw(st, ents[0], "act.rare_events", now)
    assert set(ev) == {did}
    assert [r[0] for r in ev[did]] == rare_ts
    assert [r[1] for r in ev[did]] == [100.0 + k for k in range(7)]
    assert all(r[2] == 50.0 for r in ev[did])
    for e in ents[1:]:
        assert _raw(st, e, "act.rare_events", now) is None
    assert MT.dest_prevalence(st, S, did) == pytest.approx(0.1)
    assert MT.dest_prevalence(st, S, MT.dest_id_of("erp.corp")) == pytest.approx(1.0)
    assert math.isnan(MT.dest_prevalence(st, S, 12345))
    assert MT.rare_events(st, S, ents[1], now) == {}
    arr = MT.rare_events(st, S, ents[0], now)[did]
    assert arr.shape == (7, 3) and arr.dtype == np.float64


def test_c_rare_events_outside_stream_cap_and_capped_per_dest():
    st, eng = make_store(), ActionTokenEngine()
    now = T0 + 3600
    ents = [f"10.0.1.{i}" for i in range(10)]
    batch = [_http(e, T0 + j, "/home") for e in ents[1:] for j in range(3)]
    batch += [_http(ents[0], T0 + j * 0.5, "/home") for j in range(1500)]   # one long session
    rare_ts = [T0 + 2000 + 2.0 * k for k in range(300)]
    batch += [_tls(ents[0], t, "x.beacon.net") for t in rare_ts]
    _run(eng, st, now, batch, dt=3600)
    assert len(_raw(st, ents[0], "act.stream", now)) <= 512
    ev = _raw(st, ents[0], "act.rare_events", now)[MT.dest_id_of("beacon.net")]
    assert len(ev) == MT.RARE_CAP
    assert [r[0] for r in ev] == rare_ts[-MT.RARE_CAP:]        # newest, contiguous


# ------------------------------------------------------------------ spec (d)
def test_d_silent_entity_zero_filled_without_touch():
    st, eng = make_store(), ActionTokenEngine()
    a, b = "10.0.0.1", "10.0.0.2"
    _run(eng, st, T0, [_http(a, T0 - 5, "/x"), _http(b, T0 - 4, "/x")])
    ls_b, fs_b = st.last_seen(S, b), st.first_seen(S, b)
    now = T0 + 900
    _run(eng, st, now, [_http(a, now - 5, "/x")])
    assert _raw(st, b, "act.events", now) == 0.0
    assert st.last_seen(S, b) == ls_b and st.first_seen(S, b) == fs_b
    assert _raw(st, a, "act.events", now) == 1.0
    assert _raw(st, b, "act.stream", now) is None        # absence, not an empty stream
    # the zero-filled series is the derived grid clock
    ts, v = fresh.grid(st, S, b, "act.events", now, 1800, "counter", 900)
    assert list(ts) == [T0, now] and list(v) == [1.0, 0.0]
    # an entity last seen more than 30 d ago is not zero-filled
    late = T0 + 31 * 86400
    _run(eng, st, late, [_http(a, late - 1, "/x")])
    assert _raw(st, b, "act.events", late) is None


# ------------------------------------------------------------------ spec (e)
def test_e_objs_ids_n_and_hll():
    st, eng = make_store(), ActionTokenEngine()
    e = "10.0.0.1"
    _run(eng, st, T0, [_http(e, T0 - 9 + i, f"/orders/view/{900000 + i}") for i in range(3)])
    now = T0 + 900
    ids = [str(i) for i in range(2000)]
    _run(eng, st, now, [_http(e, now - 800 + 0.3 * i, f"/orders/view/{x}")
                        for i, x in enumerate(ids)])
    objs = _raw(st, e, "act.objs", now)
    ent = objs[f"{HOST}/orders/view/{{num}}"]
    assert ent["n"] == 2000
    assert len(ent["ids"]) == MT.OBJ_IDS_CAP and ent["ids"] == ids[:MT.OBJ_IDS_CAP]
    assert isinstance(ent["hll"], bytes) and len(ent["hll"]) == 1024
    est = MT.objs_hll(ent).count()
    assert abs(est - 2000) / 2000 <= 0.05
    # small sets carry no hll; the accessor rebuilds one from the ids
    small = _raw(st, e, "act.objs", T0)[f"{HOST}/orders/view/{{num}}"]
    assert "hll" not in small or small["n"] > MT.OBJ_IDS_CAP
    assert MT.objs(st, S, e, now) is objs


def test_objs_multi_position_ids_are_joined():
    st, eng = make_store(), ActionTokenEngine()
    e = "10.0.0.1"
    _run(eng, st, T0 - 900, [_http(e, T0 - 999 + i, f"/users/9/orders/9") for i in range(3)])
    batch = [_http(e, T0 - 50 + i, f"/users/{u}/orders/{o}")
             for i, (u, o) in enumerate([(1, 10), (1, 11), (2, 10), (2, 10), (3, 12)])]
    _run(eng, st, T0, batch)
    objs = _raw(st, e, "act.objs", T0)
    assert list(objs) == [f"{HOST}/users/{{num}}/orders/{{num}}"]
    (ent,) = objs.values()
    assert ent["n"] == 4 and ent["ids"] == ["1/10", "1/11", "2/10", "3/12"]


# ------------------------------------------------------------- tokens / ids
def test_token_formats_ids_outcomes_and_stack():
    st, eng = make_store(), ActionTokenEngine()
    e = "10.0.0.1"
    ua = "python-requests/2.31.0"
    batch = [
        _http(e, T0 - 60, "/login", 200, method="POST", user_agent=ua, ttl=57, ja3=""),
        _http(e, T0 - 59, "/missing", 404),
        obs(S, e, T0 - 58, app_proto="dns", dns_qtype="TXT",
            dns_qname="a8f3k2m9x7q1w5e4r6t0y.t.evil.com", dns_rcode="NXDOMAIN", peer="10.0.0.53"),
        obs(S, e, T0 - 57, app_proto="dns", dns_qtype="A", dns_qname="www.example.org",
            dns_rcode="NOERROR"),
        _tls(e, T0 - 56, "api.cdn.example.com", up=1024, down=16384),
        obs(S, e, T0 - 55, l4_proto="tcp", dst_port=22, peer="10.1.2.3", bytes_up=4096,
            bytes_down=512),
        obs(S, e, T0 - 54, l4_proto="tcp", dst_port=4444, peer="10.1.2.3", tcp_flags="RST"),
    ]
    _run(eng, st, T0, batch)
    rows = _raw(st, e, "act.stream", T0)
    toks = [MT.token_str(st, S, t) for t in rows["token_id"]]
    assert toks[0] == "POST erp.corp /{var}|2xx"          # provisional literal
    assert toks[1] == "GET erp.corp /{var}|4xx"
    assert toks[2] == "TXT {rnd}.t.evil.com"
    assert toks[3] == "A www.example.org"
    assert toks[4] == "example.com:443 u10/d14"
    assert toks[5] == "tcp/ssh u12/d9"
    assert toks[6] == "tcp/4444 u0/d0"
    assert list(rows["outcome"]) == [2, 4, 4, 2, 2, 2, 4]
    for t, tid in zip(toks, rows["token_id"]):
        assert MT.token_id(st, S, t) == tid > 0
        assert MT.token_count(st, S, t) == MT.token_count(st, S, int(tid)) == 1.0
    assert MT.token_id(st, S, "GET nowhere /x|2xx") == 0
    assert MT.token_str(st, S, 0) == TPL.RARE_TOKEN and MT.token_str(st, "nosys", 5) == "{rare}"
    assert rows["stack_id"][0] == stack_id(stack_token("", ua, 57, 0))
    assert MT.seq_token(toks[2], rows["outcome"][2]) == "TXT {rnd}.t.evil.com|4xx"
    assert MT.seq_token(toks[0], 2) == toks[0]
    dests = [MT.dest_name(st, S, d) for d in rows["dest_id"]]
    assert dests == ["erp.corp", "erp.corp", "evil.com", "example.org", "example.com",
                     "10.1.2.0/24", "10.1.2.0/24"]
    assert _raw(st, e, "act.distinct_templates", T0) == 7.0
    assert _raw(st, e, "act.new_template_ratio", T0) == 1.0
    assert rows["up"][4] == 1024 and rows["down"][4] == 16384


def test_new_template_ratio_and_top_tokens_other():
    st, eng = make_store(), ActionTokenEngine()
    e = "10.0.0.1"
    b1 = [_http(e, T0 - 100 + i, "/a") for i in range(4)]
    _run(eng, st, T0, b1)
    assert _raw(st, e, "act.new_template_ratio", T0) == 1.0
    now = T0 + 900
    b2 = [_http(e, now - 100 + i, "/a") for i in range(3)]
    b2 += [_tls(e, now - 50 + 0.1 * i, f"h.site{i}.com") for i in range(300)]
    _run(eng, st, now, b2)
    assert _raw(st, e, "act.new_template_ratio", now) == 0.0
    toks = _raw(st, e, "act.tokens", now)
    assert len(toks) == MT.TOKENS_TOP + 1
    assert toks[f"GET {HOST} /a|2xx"] == 3
    assert sum(toks.values()) == 303 and toks[MT.OTHER] == 303 - sum(
        v for k, v in toks.items() if k != MT.OTHER)
    # no HTTP this tick -> ratio not written (stale -> NaN downstream)
    later = now + 900
    _run(eng, st, later, [_tls(e, later - 1, "h.site1.com")])
    assert _raw(st, e, "act.new_template_ratio", later) is None


def test_aggregated_observation_weights_and_ts_sample():
    st, eng = make_store(), ActionTokenEngine()
    e = "10.0.0.1"
    offs = [1.5, 2.25, 9.0, 30.0, 31.0, 120.0, 121.0, 122.5, 400.0, 401.0]
    o = _http(e, T0 - 900, "/orders/view/77", extra={
        "count": 50, "bytes_up_total": 50 * 400, "bytes_down_total": 50 * 8000,
        "ts_sample": offs})
    o2 = _tls(e, T0 - 900, "x.example.net", extra={"count": 5})        # no ts_sample
    _run(eng, st, T0, [o, o2])
    assert _raw(st, e, "act.events", T0) == 55.0
    rows = _raw(st, e, "act.stream", T0)
    assert len(rows) == 11
    http_rows = rows[rows["dest_id"] == MT.dest_id_of("erp.corp")]
    assert list(http_rows["ts"]) == [T0 - 900 + x for x in offs]
    assert np.all(http_rows["up"] == 400) and np.all(http_rows["down"] == 8000)
    assert _raw(st, e, "act.stream_frac", T0) == pytest.approx(11 / 55)
    toks = _raw(st, e, "act.tokens", T0)
    assert sum(toks.values()) == 55
    # the aggregate's 50 hits establish the literals at once (no provisional {var})
    assert toks[f"GET {HOST} /orders/view/{{num}}|2xx"] == 50
    assert MT.token_count(st, S, f"GET {HOST} /orders/view/{{num}}|2xx") == 50.0
    assert toks["example.net:443 u9/d11"] == 5
    # absolute ts_sample entries and a sample longer than count
    o3 = _http(e, T0, "/z", extra={"count": 2, "ts_sample": [T0 + 1800.5, T0 + 1801.5, 7.0]})
    _run(eng, st, T0 + 1800, [o3])
    assert list(_raw(st, e, "act.stream", T0 + 1800)["ts"]) == [T0 + 1800.5, T0 + 1801.5]


def test_stream_window_reweights_by_frac():
    st, eng = make_store(), ActionTokenEngine()
    e = "10.0.0.1"
    _run(eng, st, T0, [_http(e, T0 - 5, "/a", extra={"count": 4, "ts_sample": [0.0]})])
    _run(eng, st, T0 + 900, [_http(e, T0 + 800, "/a"), _http(e, T0 + 801, "/a")])
    ticks = MT.stream_ticks(st, S, e, T0)
    assert [(t, len(r), f) for t, r, f in ticks] == [(T0, 1, 0.25), (T0 + 900, 2, 1.0)]
    rows, w = MT.stream_window(st, S, e, T0)
    assert len(rows) == 3 and list(w) == [4.0, 1.0, 1.0]
    assert np.sum(w) == 6.0
    assert len(MT.stream_rows(st, S, e, T0 + 1)) == 0
    assert math.isnan(MT.stream_frac(st, S, e, T0 + 1))
    empty_rows, empty_w = MT.stream_window(st, S, "nobody", T0)
    assert len(empty_rows) == 0 and len(empty_w) == 0


# ------------------------------------------------------------- edge cases
def test_empty_store_and_no_observations():
    st, eng = make_store(), ActionTokenEngine()
    assert _run(eng, st, T0, []) == 0
    assert _run(eng, st, T0, None) == 0
    assert st.systems() == [] and MT.get(st, S) is None
    assert MT.vocab_size(st, S) == 0 and MT.token_id(st, S, "x") == 0
    assert math.isnan(MT.dest_prevalence(st, S, 1))
    assert MT.system_entities(st, S) == 0.0


def test_pseudo_and_probe_observations_are_dropped():
    st, eng = make_store(), ActionTokenEngine()
    batch = [_http("__system__", T0, "/x"), _http("class:r1", T0, "/x"),
             obs(S, "10.0.0.9", T0, method=AcquisitionMethod.ACTIVE_PROBE, dst_port=443,
                 l4_proto="tcp")]
    assert run_engine(eng, st, T0, observations=batch, scheduled=True) == 0
    assert st.entities(S) == [] and st.systems() == []
    rec = st.health()["raw.action_token"]
    assert rec["ok"] and rec["dropped_pseudo"] == 2 and rec["dropped_active"] == 1


def test_nan_and_garbage_inputs():
    st, eng = make_store(), ActionTokenEngine()
    e = "10.0.0.1"
    nan = float("nan")
    batch = [
        _http(e, nan, "/a", status=nan, bytes_up=nan, bytes_down=float("inf")),
        _http(e, T0 - 3, "/b", extra={"count": nan, "ts_sample": [nan, "x", 2.0]}),
        _http(e, T0 - 2, "/c", extra={"count": "bogus", "bytes_up_total": nan}),
        _http(e, T0 - 1, "/d", extra={"count": 0}),                 # zero events: skipped
        _http(e, T0 - 1, "/e", extra={"count": -3}),
        _tls(e, T0 - 1, "", up=-5, down=nan, extra={"ja4": ["unhashable"]}),
        obs(S, e, T0 - 1),                                          # nothing decoded
    ]
    _run(eng, st, T0, batch)
    rows = _raw(st, e, "act.stream", T0)
    assert _raw(st, e, "act.events", T0) == 5.0
    assert len(rows) == 5 and np.all(np.isfinite(rows["ts"]))
    assert rows["ts"][-1] == T0                                    # NaN ts -> ctx.now
    assert np.all(np.isfinite(rows["up"])) and np.all(rows["up"] >= 0)
    assert np.all(np.isfinite(rows["down"])) and np.all(rows["down"] >= 0)
    assert T0 - 3 + 2.0 in set(rows["ts"])
    assert _raw(st, e, "act.stream_frac", T0) == 1.0


def test_training_mode_learns_and_emits_no_events():
    st_t, st_l = make_store(), make_store()
    batch = [_http("10.0.0.1", T0 - i, f"/orders/view/{i}") for i in range(20)]
    _run(ActionTokenEngine(), st_t, T0, batch, training=True)
    _run(ActionTokenEngine(), st_l, T0, batch, training=False)
    assert st_t.events() == [] and st_l.events() == []
    assert MT.vocab_size(st_t, S) == MT.vocab_size(st_l, S) > 0
    assert np.array_equal(_raw(st_t, "10.0.0.1", "act.stream", T0),
                          _raw(st_l, "10.0.0.1", "act.stream", T0))


def test_cadence_switch_900_to_60():
    st, eng = make_store(), ActionTokenEngine()
    a, b = "10.0.0.1", "10.0.0.2"
    stamps = []
    now = T0
    for k in range(4):                                  # 900-s ticks, b active on k == 0
        batch = [_http(a, now - 30, "/x")] + ([_http(b, now - 20, "/x")] if k == 0 else [])
        _run(eng, st, now, batch, dt=900)
        stamps.append(now)
        now += 900
    now -= 900
    for k in range(10):                                  # then 60-s ticks
        now += 60
        _run(eng, st, now, [_http(a, now - 1, "/x")] * 2, dt=60)
        stamps.append(now)
    for t in stamps:
        assert _raw(st, a, "act.events", t) in (1.0, 2.0)
        assert _raw(st, b, "act.events", t) == (1.0 if t == T0 else 0.0)
    ts, v = fresh.grid(st, S, b, "act.events", now, 3000, "counter", 60)
    assert list(ts) == stamps[1:] and v.sum() == 0.0
    assert st.last_seen(S, b) == T0


def test_prevalence_decays_with_half_life():
    st, eng = make_store(), ActionTokenEngine()
    ents = [f"10.0.2.{i}" for i in range(5)]
    _run(eng, st, T0, [_http(e, T0 - 1, "/x") for e in ents])
    assert MT.system_entities(st, S) == pytest.approx(5.0)
    week = T0 + MT.PREV_HALF_LIFE_S
    assert MT.system_entities(st, S, now=week) == pytest.approx(2.5)
    # one entity keeps going for a week: it is back at weight 1, the rest decay
    _run(eng, st, week, [_http(ents[0], week - 1, "/x")])
    assert MT.system_entities(st, S) == pytest.approx(1.0 + 4 * 0.5)
    did = MT.dest_id_of(HOST)
    assert MT.dest_entities(st, S, did) == pytest.approx(3.0)
    # after 9 weeks of only ents[0] the others are pruned by the sweep
    far = T0 + 9 * MT.PREV_HALF_LIFE_S
    _run(eng, st, far, [_http(ents[0], far - 1, "/x")])
    assert MT.system_entities(st, S) == pytest.approx(1.0)
    assert MT.dest_prevalence(st, S, did) == pytest.approx(1.0)


def test_model_serialisation_round_trip():
    st, eng = make_store(), ActionTokenEngine()
    batch = [_http(f"10.0.0.{i % 3}", T0 - i, f"/orders/view/{i}") for i in range(30)]
    batch += [_tls("10.0.0.1", T0 - 1, "a.example.com")]
    _run(eng, st, T0, batch)
    m = MT.get(st, S)
    assert m["version"] == 1 == MT.version(st, S)
    assert st.model_version(S, "__system__", MT.MODEL) == 1
    d = json.loads(json.dumps(MT.to_dict(m)))
    m2 = MT.from_dict(d)
    assert m2["templater"].vocab == m["templater"].vocab
    assert m2["prev"]["dests"] == m["prev"]["dests"]
    assert m2["prev"]["ents"] == m["prev"]["ents"]
    assert m2["prev"]["ent_sum"] == m["prev"]["ent_sum"]
    assert MT.from_dict(m) is m and MT.from_dict(None)["version"] == 0
    # a restored (JSON-form) model is picked up and learning continues
    st.put_model(S, "__system__", MT.MODEL, d)
    _run(eng, st, T0 + 900, [_http("10.0.0.1", T0 + 800, "/orders/view/5")])
    assert isinstance(MT.get(st, S)["templater"], TPL.Templater)
    assert MT.version(st, S) == 2


def test_template_key_and_outcome_helpers():
    assert MT.template_key("GET h /a/{num}|2xx") == "GET h /a/{num}"
    assert MT.template_key("example.com:443 u10/d14") == "example.com:443"
    assert MT.template_key("tcp/ssh u12/d9") == "tcp/ssh"
    assert MT.template_key("A www.example.org") == "A www.example.org"
    assert [MT.outcome_class(c) for c in (0, 1, 2, 3, 4, 5, 9, "x")] == \
        ["0xx", "1xx", "2xx", "3xx", "4xx", "5xx", "0xx", "0xx"]
    assert MT.dest_name_of("2001:db8::1") == "2001:db8::/64"
    assert MT.dest_name_of("") == "" and MT.dest_id_of("") == 0
    assert 0 < MT.dest_id_of("example.com") <= 0x7FFFFFFF


# ------------------------------------------------------------------ perf
def test_perf_2000_observations():
    rng = random.Random(11)
    ents = [f"10.0.3.{i}" for i in range(40)]
    paths = ["/orders/view/{}", "/api/v1/items/{}", "/home", "/search?q={}&page=2",
             "/static/app.js", "/report/daily"]

    def batch(now):
        out = []
        for e in ents:
            t = now - 800
            for _ in range(50):
                t += rng.uniform(0.2, 20)
                p = rng.choice(paths).format(rng.randrange(10 ** 6))
                out.append(_http(e, t, p, rng.choice([200, 200, 200, 404]),
                                 user_agent="Mozilla/5.0 Chrome/126.0", ttl=128))
        return out

    st, eng = make_store(), ActionTokenEngine()
    for k in range(3):                                  # warm the vocabulary
        _run(eng, st, T0 + k * 900, batch(T0 + k * 900))
    b = batch(T0 + 3 * 900)
    t0 = time.perf_counter()
    _run(eng, st, T0 + 3 * 900, b)
    dt = time.perf_counter() - t0
    assert dt < 0.25, f"{dt * 1e3:.1f} ms for 2000 observations"


# ------------------------------------------------------- internals / guards
def _greedy_reference(ts, s, e, now):
    """Sequential form of _cap_index (same seeded draws, block-by-block loop)."""
    from app.engines.behavior.lib.combine import seeded_uniform
    cap, n = MT.STREAM_CAP, len(ts)
    cut = [i for i in range(1, n) if ts[i] - ts[i - 1] > MT.SESSION_GAP_S]
    starts, ends = [0] + cut, cut + [n]
    rng = np.random.default_rng(int(seeded_uniform(s, e, float(now), "act.stream") * 2.0 ** 53))
    blocks = []
    for a, b in zip(starts, ends):
        if b - a > cap:
            a += int(rng.integers(-(-(b - a) // cap))) * cap
            b = min(a + cap, b)
        blocks.append((a, b))
    budget, keep = cap, []
    for i in rng.permutation(len(blocks)).tolist():
        a, b = blocks[i]
        if b - a <= budget:
            keep.append(i)
            budget -= b - a
    return [k for i in sorted(keep) for k in range(*blocks[i])]


@pytest.mark.parametrize("seed", range(12))
def test_cap_index_is_exact_greedy_and_keeps_gaps_true(seed):
    from app.engines.raw.action_token import _cap_index
    rng = random.Random(seed)
    t, ts = 0.0, []
    while len(ts) < 520 + 150 * seed:
        size = rng.choice([1, 2, 5, 40, 90, 300, 700]) if seed % 3 else rng.randint(1, 60)
        for _ in range(size):
            ts.append(t)
            t += rng.uniform(0.0, 29.0)
        t += rng.uniform(30.5, 500.0)
    ts = np.array(ts)
    idx = _cap_index(ts, S, f"e{seed}", T0)
    assert idx.tolist() == _greedy_reference(ts, S, f"e{seed}", T0)
    assert 0 < len(idx) <= MT.STREAM_CAP and np.all(np.diff(idx) > 0)
    kept = ts[idx]
    jumps = np.diff(idx) > 1                          # a dropped row in between
    assert np.all(np.diff(kept)[jumps] > MT.SESSION_GAP_S)


def test_outcome_codes_and_method_strings():
    st, eng = make_store(), ActionTokenEngine()
    e = "10.0.0.1"
    q = dict(app_proto="dns", dns_qtype="A", dns_qname="www.example.org")
    batch = [obs(S, e, T0 - 5, dns_rcode=0, **q), obs(S, e, T0 - 4, dns_rcode="SERVFAIL", **q),
             obs(S, e, T0 - 3, dns_rcode=3.0, **q), obs(S, e, T0 - 2, dns_rcode="", **q),
             obs(S, e, T0 - 1, dns_rcode=float("inf"), **q),
             obs(S, e, T0, method="active_probe", l4_proto="tcp", dst_port=22)]
    _run(eng, st, T0, batch)
    rows = _raw(st, e, "act.stream", T0)
    assert list(rows["outcome"]) == [2, 5, 4, 0, 5]
    assert eng.health_record()["dropped_active"] == 1


def test_clock_stepping_back_is_safe():
    st, eng = make_store(), ActionTokenEngine()
    ents = [f"10.0.4.{i}" for i in range(4)]
    later = T0 + 5 * 86400
    _run(eng, st, later, [_http(e, later - 1, "/x") for e in ents])
    _run(eng, st, T0, [_http(e, T0 - 1, "/x") for e in ents[:2]])     # replay from earlier
    n = MT.system_entities(st, S)
    assert 0 < n <= 4.0 + 1e-9
    assert 0.0 < MT.dest_prevalence(st, S, MT.dest_id_of(HOST)) <= 1.0


def test_chunk_expansion_paths_agree(monkeypatch):
    """_rows_array expands aggregated ts_sample chunks in Python when small and
    with numpy when large; both must give identical rows."""
    from app.engines.raw import action_token as AT
    rng = random.Random(5)
    junk = [float("nan"), float("inf"), "x", None, [1, 2], "2.5", True, 10 ** 400]
    acc = AT._Acc()
    acc.rows.append((T0, 1, 2, 10.0, 20.0, 3, 4))
    for j in range(40):
        head = [rng.uniform(-5, 900) for _ in range(rng.randint(1, 20))]
        if j % 5 == 0:
            head[0] = T0 + 1234.5                       # absolute epoch
        if j % 7 == 0:
            head = [rng.choice(junk) for _ in range(3)]  # nothing usable -> one row at t0
        if j % 3 == 0:
            head.append(rng.choice(junk))
        acc.chunks.append((head, T0 - 900 + j, j + 1, 2, 100.0 + j, 50.0, 7, 9))
    monkeypatch.setattr(AT, "_PY_EXPAND_MAX", 10 ** 9)
    py = AT._rows_array(acc)
    monkeypatch.setattr(AT, "_PY_EXPAND_MAX", 0)
    vec = AT._rows_array(acc)
    assert py.dtype == vec.dtype == MT.STREAM_DTYPE
    assert len(py) == len(vec) and np.all(np.isfinite(py["ts"]))
    assert py.tobytes() == vec.tobytes()
    assert set(py["token_id"]) == set(range(1, 41)) | {1}     # every record keeps >= 1 row
