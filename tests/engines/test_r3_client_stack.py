"""R3 ClientStackEngine (engines/raw/client_stack.py).

Spec unit tests (docs/lib3/engines.md, R3):
  (a) two JA3 strings that differ only in extension order map to one ja3n;
  (b) a python-requests UA with TTL 57 -> token '...|python-requests/2|linux|64|...';
  (c) two stacks with interleaved obs.ts inside one 900-s tick -> stack_events
      whose [first_ts, last_ts] intervals overlap.
Plus: the 5-minute concurrency test gives the same answer at 60 / 900 / 3600 s,
aggregated observations (count, bytes totals, ts_sample), JA4, component sets
and OS/TTL pairs, top-64 cap, episode cap, pseudo / probe / blank observations,
empty store, silent entity, NaN inputs, training mode, cadence 900 -> 60,
JSON-safety, the Python / numpy episode paths agreeing, and perf.
"""
from __future__ import annotations

import json
import math
import random
import time

import numpy as np
import pytest

from helpers import T0, make_store, obs, run_engine

from app.engines.behavior.lib import sketch
from app.engines.behavior.lib import stack as S
from app.engines.raw import client_stack as CS
from app.engines.raw.client_stack import ClientStackEngine
from app.models.schema import AcquisitionMethod

SYS = "erp"
E = "10.20.1.11"
NAN = float("nan")

UA_CHROME_WIN = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                 "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
UA_REQUESTS = "python-requests/2.31.0"
CH_CIPHERS = [4865, 4866, 4867, 49195, 49199, 49196, 49200, 52393, 52392, 49171, 49172,
              156, 157, 47, 53]
CH_EXTS = [0, 23, 65281, 10, 11, 35, 16, 5, 13, 18, 51, 45, 43, 27, 17513, 21]
CH_CURVES = [29, 23, 24]
PY_JA3 = "771,4866-4867-4865-49196-49200,0-11-10-35-22-23-13-43-45-51,29-23-30-25-24,0-1-2"
NAMES = ("client.stack_set", "client.stack_events", "client.ua_set", "client.ja3n_set",
         "client.ttl_set", "client.os_ua_ttl_pairs")


def j(*lists) -> str:
    return ",".join("-".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x)
                    for x in lists)


CHROME_JA3 = j(771, CH_CIPHERS, CH_EXTS, CH_CURVES, [0])


def _chrome(e, ts, **kw):
    return obs(SYS, e, ts, app_proto="https", ja3=kw.pop("ja3", CHROME_JA3),
               user_agent=kw.pop("user_agent", UA_CHROME_WIN), ttl=kw.pop("ttl", 120),
               win_size=kw.pop("win_size", 64240), bytes_up=kw.pop("bytes_up", 500),
               bytes_down=kw.pop("bytes_down", 4000), **kw)


def _pyreq(e, ts, **kw):
    return obs(SYS, e, ts, app_proto="https", ja3=kw.pop("ja3", PY_JA3),
               user_agent=kw.pop("user_agent", UA_REQUESTS), ttl=kw.pop("ttl", 57),
               win_size=kw.pop("win_size", 29200), bytes_up=kw.pop("bytes_up", 300),
               bytes_down=kw.pop("bytes_down", 900), **kw)


def _run(eng, st, now, observations, dt=900.0, training=False):
    return run_engine(eng, st, now, training=training, dt=dt, observations=observations)


def _raw(st, e, name, ts, s=SYS):
    m = st.latest_raw_at(s, e, name, ts)
    return None if m is None else m.value


def _tok(ja3, ua, ttl, win, ja4=None):
    return S.stack_token(ja3, ua, ttl, win, ja4)


def _rows_of(rows, sid):
    return [r for r in rows if r[0] == sid]


def _concurrent(rows_a, rows_b, win=300.0) -> bool:
    """B09's test: some A and B episode overlap or are within `win` seconds."""
    return any(a[1] <= b[2] + win and b[1] <= a[2] + win for a in rows_a for b in rows_b)


# ------------------------------------------------------------------ spec (a)
def test_a_extension_order_maps_to_one_ja3n():
    rng = random.Random(3)
    exts = CH_EXTS[:]
    rng.shuffle(exts)
    ja3_1 = j(771, CH_CIPHERS, CH_EXTS, CH_CURVES, [0])
    ja3_2 = j(771, CH_CIPHERS, exts, CH_CURVES, [0])
    ja3_3 = j(771, [0x3A3A] + CH_CIPHERS, [0x8A8A] + exts[::-1], [0x2A2A] + CH_CURVES, [0])
    assert ja3_1 != ja3_2
    st, eng = make_store(), ClientStackEngine()
    now = T0 + 900
    _run(eng, st, now, [_chrome(E, now - 800, ja3=ja3_1), _chrome(E, now - 700, ja3=ja3_2),
                        _chrome(E, now - 600, ja3=ja3_3)])
    j3 = _raw(st, E, "client.ja3n_set", now)
    assert j3 == {S.ja3n(ja3_1): 3}
    ss = _raw(st, E, "client.stack_set", now)
    assert list(ss) == [_tok(ja3_1, UA_CHROME_WIN, 120, 64240)]
    assert ss[next(iter(ss))]["n"] == 3


# ------------------------------------------------------------------ spec (b)
def test_b_python_requests_ttl57_token():
    st, eng = make_store(), ClientStackEngine()
    now = T0 + 900
    _run(eng, st, now, [_pyreq(E, now - 100)])
    ss = _raw(st, E, "client.stack_set", now)
    (tok,) = ss
    assert "|python-requests/2|linux|64|" in tok
    assert tok.split("|")[1:4] == ["python-requests/2", "linux", "64"]
    assert tok == _tok(PY_JA3, UA_REQUESTS, 57, 29200)
    assert _raw(st, E, "client.ua_set", now) == {"python-requests/2": 1}
    assert _raw(st, E, "client.ttl_set", now) == {"64": 1}
    # the UA declares no OS: '?' (not the TTL-implied 'linux') for B09's check
    assert _raw(st, E, "client.os_ua_ttl_pairs", now) == {"?|python-requests|64": 1}


# ------------------------------------------------------------------ spec (c)
def test_c_interleaved_stacks_overlap_in_one_tick():
    st, eng = make_store(), ClientStackEngine()
    now = T0 + 900
    t0 = now - 900
    batch = []
    for i in range(12):                                # A at +10, +70, ...; B at +40, +100, ...
        batch.append(_chrome(E, t0 + 10 + 60 * i))
        batch.append(_pyreq(E, t0 + 40 + 60 * i))
    random.Random(1).shuffle(batch)                    # arrival order must not matter
    _run(eng, st, now, batch, dt=900)
    rows = _raw(st, E, "client.stack_events", now)
    sa = S.stack_id(_tok(CHROME_JA3, UA_CHROME_WIN, 120, 64240))
    sb = S.stack_id(_tok(PY_JA3, UA_REQUESTS, 57, 29200))
    (ra,), (rb,) = _rows_of(rows, sa), _rows_of(rows, sb)
    assert ra == [sa, t0 + 10, t0 + 670, 12]
    assert rb == [sb, t0 + 40, t0 + 700, 12]
    assert ra[1] <= rb[2] and rb[1] <= ra[2]           # [first, last] intervals overlap
    ss = _raw(st, E, "client.stack_set", now)
    assert {S.stack_id(t) for t in ss} == {sa, sb}
    for t, v in ss.items():
        assert (v["first_ts"], v["last_ts"]) == tuple(_rows_of(rows, S.stack_id(t))[0][1:3])
    assert rows == sorted(rows, key=lambda r: (r[1], r[0]))


# ------------------------------------------------- concurrency at any cadence
PATTERNS = {
    # A and B alternate 400 s apart: never within 5 min of each other
    "alternate_400": ([0, 800, 1600, 2400], [400, 1200, 2000]),
    # A every 60 s until 600, B once at 850: 250 s after A's last event
    "near_250": (list(range(0, 601, 60)), [850]),
    # replacement: A until 1000, B from 1400
    "replace": (list(range(0, 1001, 100)), list(range(1400, 3500, 100))),
    # interleaved every 30 s over 50 minutes (NAT / impersonation signature)
    "interleave": (list(range(5, 3000, 60)), list(range(35, 3000, 60))),
    # A at the start and end of the hour, B in the middle (an envelope would overlap)
    "sandwich": ([10, 20, 3550, 3590], [1800]),
}


def _brute(ta, tb, win=300.0) -> bool:
    return any(abs(a - b) <= win for a in ta for b in tb)


@pytest.mark.parametrize("name", sorted(PATTERNS))
def test_concurrency_is_cadence_invariant(name):
    ta, tb = PATTERNS[name]
    sa = S.stack_id(_tok(CHROME_JA3, UA_CHROME_WIN, 120, 64240))
    sb = S.stack_id(_tok(PY_JA3, UA_REQUESTS, 57, 29200))
    base = T0 - (T0 % 3600) + 3600
    answers = set()
    for dt in (60.0, 300.0, 900.0, 3600.0):
        st, eng = make_store(), ClientStackEngine()
        events = [(base + x + 0.5, _chrome) for x in ta] + [(base + x + 0.5, _pyreq) for x in tb]
        rows = []
        for k in range(int(3600 / dt)):
            now = base + (k + 1) * dt
            batch = [f(E, t) for t, f in events if now - dt < t <= now]
            if not batch:
                continue
            _run(eng, st, now, batch, dt=dt)
            rows += _raw(st, E, "client.stack_events", now)
        ra, rb = _rows_of(rows, sa), _rows_of(rows, sb)
        assert sum(r[3] for r in ra) == len(ta) and sum(r[3] for r in rb) == len(tb)
        answers.add(_concurrent(ra, rb))
    assert answers == {_brute(ta, tb)}


# ------------------------------------------------------------ aggregated obs
def test_aggregated_record_ts_sample_and_bytes():
    st, eng = make_store(), ClientStackEngine()
    now = T0 + 3600
    t0 = now - 3600
    offs = [0, 10, 20, 30, 40, 1000, 1010, 1020, 1030, 1040]    # two episodes
    o = _chrome(E, t0, extra={"count": 53, "bytes_up_total": 5000, "bytes_down_total": 90000,
                              "ts_sample": offs})
    _run(eng, st, now, [o], dt=3600)
    ss = _raw(st, E, "client.stack_set", now)
    (v,) = ss.values()
    assert v == {"n": 53, "bytes": 95000.0, "first_ts": t0, "last_ts": t0 + 1040}
    rows = _raw(st, E, "client.stack_events", now)
    assert [r[1:] for r in rows] == [[t0, t0 + 40, 28], [t0 + 1000, t0 + 1040, 25]]
    assert isinstance(rows[0][3], int)
    assert _raw(st, E, "client.ua_set", now) == {"chrome/126": 53}


def test_aggregated_record_sample_edge_cases():
    st, eng = make_store(), ClientStackEngine()
    now = T0 + 900
    t0 = now - 900
    recs = [
        # absolute epochs, NaN and junk dropped; only the first w=3 entries are used
        _chrome(E, t0, extra={"count": 3, "ts_sample": [t0 + 5, NAN, "x", t0 + 7, t0 + 9]}),
        # no valid entry: the whole weight at obs.ts
        _pyreq(E, t0 + 100, extra={"count": 4, "ts_sample": [NAN, None]}),
        # no ts_sample: one event of weight 7 at obs.ts; no totals: w * bytes
        _pyreq(E, t0 + 110, extra={"count": 7}),
        # count <= 0 is no event; NaN / junk count is one event
        _pyreq(E, t0 + 120, extra={"count": 0}),
        _pyreq(E, t0 + 130, extra={"count": -3}),
        _pyreq(E, t0 + 140, extra={"count": NAN}),
        _pyreq(E, t0 + 150, extra={"count": "junk"}),
        # an empty / non-list ts_sample is ignored
        _pyreq(E, t0 + 160, extra={"count": 2, "ts_sample": []}),
        _pyreq(E, t0 + 170, extra={"count": 2, "ts_sample": "12"}),
    ]
    _run(eng, st, now, recs)
    ss = _raw(st, E, "client.stack_set", now)
    rows = _raw(st, E, "client.stack_events", now)
    ta = _tok(CHROME_JA3, UA_CHROME_WIN, 120, 64240)
    tb = _tok(PY_JA3, UA_REQUESTS, 57, 29200)
    assert ss[ta]["n"] == 3 and (ss[ta]["first_ts"], ss[ta]["last_ts"]) == (t0 + 5, t0 + 5)
    # only [t0+5, nan, 'x'] were read (first w = 3 entries): one valid -> weight 3
    assert _rows_of(rows, S.stack_id(ta)) == [[S.stack_id(ta), t0 + 5, t0 + 5, 3]]
    assert ss[tb]["n"] == 4 + 7 + 1 + 1 + 2 + 2
    assert ss[tb]["bytes"] == (4 + 7 + 1 + 1 + 2 + 2) * 1200.0
    assert _rows_of(rows, S.stack_id(tb)) == [[S.stack_id(tb), t0 + 100, t0 + 170, 17]]


def test_weight_split_over_samples_is_exact():
    st, eng = make_store(), ClientStackEngine()
    now = T0 + 86400
    t0 = now - 86400
    for w, k in ((5, 5), (63, 5), (64, 7), (65, 64), (1000, 3)):
        st = make_store()
        offs = [400.0 * i for i in range(k)]                    # every sample its own episode
        _run(eng, st, now, [_pyreq(E, t0, extra={"count": w, "ts_sample": offs})], dt=86400)
        ns = [r[3] for r in _raw(st, E, "client.stack_events", now)]
        if k > CS.EPISODES_CAP:
            assert len(ns) == CS.EPISODES_CAP and sum(ns) == w
            continue
        base, rem = divmod(w, k)
        assert ns == [base + 1] * rem + [base] * (k - rem)


# ------------------------------------------------------ reference comparison
def _ref_events(o, now):
    """Independent reimplementation of the documented event-time rules."""
    def fl(x):
        try:
            return float(x)
        except (TypeError, ValueError, OverflowError):
            return NAN
    t0 = fl(o.ts)
    t0 = t0 if math.isfinite(t0) else now
    ex = o.extra or {}
    if not ex:
        return [(t0, 1)]
    c = ex.get("count", 1)
    cf = fl(c)
    w = 1 if not math.isfinite(cf) else (int(cf) if cf >= 1 else 0)
    if w <= 0:
        return []
    smp = ex.get("ts_sample")
    if not isinstance(smp, (list, tuple)) or not smp:
        return [(t0, w)]
    ts = []
    for x in list(smp)[:min(len(smp), 64, w)]:
        v = fl(x)
        if math.isfinite(v):
            ts.append(t0 + v if abs(v) < 1e8 else v)
    if not ts:
        return [(t0, w)]
    base, rem = divmod(w, len(ts))
    return [(t, base + (i < rem)) for i, t in enumerate(ts)]


def _ref_rows(observations, now):
    per = {}
    for o in observations:
        tok = S.stack_token(o.ja3, o.user_agent, o.ttl, o.win_size, (o.extra or {}).get("ja4"))
        per.setdefault(tok, []).extend(_ref_events(o, now))
    rows = []
    for tok, evs in per.items():
        evs = sorted(evs, key=lambda x: x[0])
        eps = [[evs[0][0], evs[0][0], 0]]
        for t, w in evs:
            if t - eps[-1][1] > 300.0:
                eps.append([t, t, 0])
            eps[-1][1] = t
            eps[-1][2] += w
        while len(eps) > CS.EPISODES_CAP:                      # merge the narrowest gap
            g = min(range(len(eps) - 1), key=lambda i: (eps[i + 1][0] - eps[i][1], -i))
            a, b = eps[g], eps.pop(g + 1)
            a[1], a[2] = b[1], a[2] + b[2]
        rows += [[S.stack_id(tok), f, la, n] for f, la, n in eps if n > 0]
    return sorted(rows, key=lambda r: (r[1], r[0]))


def test_matches_reference_on_random_ticks():
    rng = random.Random(11)
    fps = [dict(ja3=CHROME_JA3, user_agent=UA_CHROME_WIN, ttl=120, win_size=64240),
           dict(ja3=PY_JA3, user_agent=UA_REQUESTS, ttl=57, win_size=29200),
           dict(user_agent="curl/8.4.0", ttl=250),
           dict(ja3=CHROME_JA3, ttl=64)]
    for trial in range(40):
        dt = rng.choice([60.0, 900.0, 3600.0, 86400.0])
        now = T0 + dt * (trial + 1)
        batch = []
        for _ in range(rng.randrange(1, 60)):
            f = rng.choice(fps)
            t = now - rng.random() * dt
            kind = rng.random()
            if kind < 0.5:
                batch.append(obs(SYS, E, t, **f))
            elif kind < 0.7:
                batch.append(obs(SYS, E, t, extra={"count": rng.randrange(-1, 9)}, **f))
            else:
                w = rng.randrange(1, 120)
                smp = [rng.choice([rng.random() * dt, -rng.random() * 50, NAN, "bad",
                                   now - rng.random() * dt])
                       for _ in range(rng.randrange(0, 80))]
                batch.append(obs(SYS, E, t, extra={"count": w, "ts_sample": smp}, **f))
        st, eng = make_store(), ClientStackEngine()
        _run(eng, st, now, batch, dt=dt)
        want = _ref_rows(batch, now)
        got = _raw(st, E, "client.stack_events", now) or []
        assert got == want, trial
        if got:
            ss = _raw(st, E, "client.stack_set", now)
            for tok, v in ss.items():
                mine = _rows_of(got, S.stack_id(tok))
                assert v["n"] == sum(r[3] for r in mine)
                assert (v["first_ts"], v["last_ts"]) == (mine[0][1], mine[-1][2])


# ------------------------------------------------------------ components
def test_ja4_takes_precedence():
    st, eng = make_store(), ClientStackEngine()
    now = T0 + 900
    ja4 = "t13d1516h2_8daaf6152771_02713d6af862"
    _run(eng, st, now, [_chrome(E, now - 10, extra={"ja4": ja4}), _chrome(E, now - 5)])
    ss = _raw(st, E, "client.stack_set", now)
    assert set(ss) == {_tok(CHROME_JA3, UA_CHROME_WIN, 120, 64240, ja4),
                       _tok(CHROME_JA3, UA_CHROME_WIN, 120, 64240)}
    assert _raw(st, E, "client.ja3n_set", now) == {"ja4:" + ja4: 1, S.ja3n(CHROME_JA3): 1}


def test_os_ua_ttl_pairs_feed_the_consistency_check():
    st, eng = make_store(), ClientStackEngine()
    now = T0 + 900
    # a copied Windows Chrome UA from a Linux stack (TTL 57 -> class 64), and a genuine one
    _run(eng, st, now, [_chrome(E, now - 30, ttl=57), _chrome(E, now - 20, ttl=120),
                        _chrome(E, now - 10, ttl=120)])
    pairs = _raw(st, E, "client.os_ua_ttl_pairs", now)
    assert pairs == {"win|chrome|64": 1, "win|chrome|128": 2}
    verdict = {k: S.os_ttl_consistent(k.split("|")[0], k.split("|")[2]) for k in pairs}
    assert verdict == {"win|chrome|64": False, "win|chrome|128": True}
    assert _raw(st, E, "client.ttl_set", now) == {"64": 1, "128": 2}


def test_partial_and_blank_observations():
    st, eng = make_store(), ClientStackEngine()
    now = T0 + 900
    dns = obs(SYS, E, now - 50, app_proto="dns", dns_qname="erp.corp", ttl=60)     # TTL only
    blank = obs(SYS, E, now - 40, app_proto="dns", dns_qname="erp.corp")          # nothing
    tls = obs(SYS, E, now - 30, app_proto="tls", ja3=CHROME_JA3, win_size=65535)  # no UA / TTL
    only_blank = obs(SYS, "10.20.1.12", now - 20, dns_qname="x.corp")
    _run(eng, st, now, [dns, blank, tls, only_blank])
    ss = _raw(st, E, "client.stack_set", now)
    assert set(ss) == {"-|none/0|linux|64|w0", S.ja3n(CHROME_JA3) + "|none/0|?|0|w16"}
    assert sum(v["n"] for v in ss.values()) == 2                  # the blank one is not counted
    assert _raw(st, E, "client.ua_set", now) is None              # no UA seen: no set
    assert _raw(st, E, "client.os_ua_ttl_pairs", now) is None     # needs UA and TTL
    assert _raw(st, E, "client.ttl_set", now) == {"64": 1}
    assert _raw(st, E, "client.ja3n_set", now) == {S.ja3n(CHROME_JA3): 1}
    for name in NAMES:                                            # blank-only entity: nothing
        assert _raw(st, "10.20.1.12", name, now) is None
    assert eng.health_record()["unfingerprinted"] == 2


def test_top64_cap_and_other():
    st, eng = make_store(), ClientStackEngine()
    now = T0 + 900
    batch = []
    for i in range(100):                                          # 100 distinct UA majors
        for r in range(1 + (i < 10)):
            batch.append(_pyreq(E, now - 800 + i + r * 0.1, user_agent=f"bot-agent/{i + 1}"))
    _run(eng, st, now, batch)
    total = len(batch)
    ss = _raw(st, E, "client.stack_set", now)
    assert len(ss) == 65 and CS.OTHER in ss
    assert sum(v["n"] for v in ss.values()) == total
    ranked = sorted({}.fromkeys(_tok(PY_JA3, f"bot-agent/{i + 1}", 57, 29200)
                                for i in range(100)),
                    key=lambda t: (-(2 if int(t.split("|")[1].split("/")[1]) <= 10 else 1), t))
    assert list(ss)[:64] == ranked[:64]                           # top by n, ties by token
    dropped = [int(t.split("|")[1].split("/")[1]) - 1 for t in ranked[64:]]
    assert ss[CS.OTHER] == {"n": 36, "bytes": 36 * 1200.0,
                            "first_ts": now - 800 + min(dropped),
                            "last_ts": now - 800 + max(dropped)}
    kept = {S.stack_id(t) for t in ss if t != CS.OTHER}
    rows = _raw(st, E, "client.stack_events", now)
    assert {r[0] for r in rows} == kept
    ua = _raw(st, E, "client.ua_set", now)
    assert len(ua) == 65 and sum(ua.values()) == total
    assert _raw(st, E, "client.ttl_set", now) == {"64": total}
    assert _raw(st, E, "client.os_ua_ttl_pairs", now) == {"?|other|64": total}  # family only
    blk = sketch.sketch_block(ss, "client.stack_set")             # the B01 sketch reads it
    assert abs(float(np.linalg.norm(blk)) - 1.0) < 1e-9


def test_episode_cap_merges_closest_gaps():
    st, eng = make_store(), ClientStackEngine()
    now = T0 + 86400
    t0 = now - 86400
    ts = [t0 + 400.0 * i + (37.0 if i % 7 == 0 else 0.0) for i in range(50)]
    _run(eng, st, now, [_pyreq(E, t) for t in ts], dt=86400)
    rows = _raw(st, E, "client.stack_events", now)
    assert len(rows) == CS.EPISODES_CAP
    assert sum(r[3] for r in rows) == 50
    assert all(any(r[1] <= t <= r[2] for r in rows) for t in ts)
    assert all(a[2] < b[1] for a, b in zip(rows, rows[1:]))       # disjoint, sorted
    # the kept cuts are the widest gaps (437 s ones), the 363 s gaps were merged
    gaps = [b[1] - a[2] for a, b in zip(rows, rows[1:])]
    assert min(gaps) >= 400.0


# ------------------------------------------------------------ edge cases
def test_empty_store_and_no_observations():
    st, eng = make_store(), ClientStackEngine()
    assert _run(eng, st, T0, None) == 0
    assert _run(eng, st, T0 + 900, []) == 0
    assert st.systems() == [] and st.entities(SYS) == []


def test_pseudo_entities_and_probes_are_dropped():
    st, eng = make_store(), ClientStackEngine()
    now = T0 + 900
    batch = [_chrome("__system__", now - 10), _chrome("class:r1", now - 10),
             _chrome("", now - 10),
             _chrome(E, now - 10, method=AcquisitionMethod.ACTIVE_PROBE),
             _chrome(E, now - 10, method=AcquisitionMethod.ACTIVE_TLS)]
    assert _run(eng, st, now, batch) == 0
    assert st.entities(SYS) == [] and st.entities(SYS, include_pseudo=True) == []
    rec = eng.health_record()
    assert rec["dropped_pseudo"] == 3 and rec["dropped_active"] == 2


def test_silent_entity_gets_nothing_and_last_seen_is_kept():
    st, eng = make_store(), ClientStackEngine()
    t1, t2 = T0 + 900, T0 + 1800
    _run(eng, st, t1, [_chrome(E, t1 - 100), _pyreq("10.20.1.12", t1 - 50)])
    assert st.last_seen(SYS, "10.20.1.12") == t1
    _run(eng, st, t2, [_chrome(E, t2 - 100)])
    for name in NAMES[:3]:
        assert st.latest_fresh(SYS, "10.20.1.12", name, t2) is None
        assert st.latest_fresh(SYS, E, name, t2) is not None
    assert st.last_seen(SYS, "10.20.1.12") == t1 and st.last_seen(SYS, E) == t2


def test_nan_and_junk_inputs():
    st, eng = make_store(), ClientStackEngine()
    now = T0 + 900
    batch = [
        obs(SYS, E, NAN, ja3="garbage,,x", user_agent=UA_REQUESTS, ttl=NAN, win_size=NAN,
            bytes_up=NAN, bytes_down=-5),
        obs(SYS, E, float("inf"), user_agent="", ttl=-1, win_size=float("inf"), bytes_up=10),
        obs(SYS, E, "junk", ja3=None, user_agent=None, ttl=300, extra={"ja4": ["a", "b"]}),
        obs(SYS, E, now - 5, ja3=CHROME_JA3, extra={"count": 2, "bytes_up_total": NAN,
                                                     "ts_sample": [float("inf"), "1"]}),
    ]
    _run(eng, st, now, batch)
    ss = _raw(st, E, "client.stack_set", now)
    for tok, v in ss.items():
        assert v["n"] >= 1 and math.isfinite(v["bytes"]) and v["bytes"] >= 0
        assert math.isfinite(v["first_ts"]) and math.isfinite(v["last_ts"])
    assert "-|python-requests/2|?|0|w0" in ss                     # junk JA3 / TTL / window
    assert ss["-|python-requests/2|?|0|w0"]["first_ts"] == now    # NaN ts -> ctx.now
    assert "-|none/0|?|0|w20" in ss                               # ttl -1 unknown, win inf
    assert "-|none/0|net|255|w0" in ss                            # ttl 300 -> 255
    rows = _raw(st, E, "client.stack_events", now)
    assert all(math.isfinite(x) for r in rows for x in r)
    assert sum(r[3] for r in rows) == sum(v["n"] for v in ss.values()) == 5


def test_training_mode_writes_metrics_but_no_events():
    st, eng = make_store(), ClientStackEngine()
    now = T0 + 900
    n = _run(eng, st, now, [_chrome(E, now - 10), _pyreq(E, now - 5)], training=True)
    assert n == 6
    assert _raw(st, E, "client.stack_set", now) is not None
    assert st.events() == []


def test_cadence_900_then_60():
    st, eng = make_store(), ClientStackEngine()
    now = T0 + 900
    _run(eng, st, now, [_chrome(E, now - 890), _chrome(E, now - 10)], dt=900)
    rows = _raw(st, E, "client.stack_events", now)
    assert [r[1:] for r in rows] == [[now - 890, now - 890, 1], [now - 10, now - 10, 1]]
    for k in range(1, 6):
        t = now + 60 * k
        _run(eng, st, t, [_chrome(E, t - 50), _pyreq(E, t - 20)], dt=60)
        rows = _raw(st, E, "client.stack_events", t)
        assert sorted(r[1] for r in rows) == [t - 50, t - 20]
        assert all(t - 60 < r[1] <= r[2] <= t for r in rows)
        ss = _raw(st, E, "client.stack_set", t)
        assert sum(v["n"] for v in ss.values()) == 2
    # the previous tick's outputs are untouched
    assert len(st.raw_series(SYS, E, "client.stack_set")) == 6


def test_outputs_are_json_safe_plain_python():
    st, eng = make_store(), ClientStackEngine()
    now = T0 + 900
    _run(eng, st, now, [_chrome(E, now - 10, extra={"count": np.int64(3),
                                                   "ts_sample": np.array([0.0, 1.5])}),
                        _pyreq(E, np.float64(now - 5))])
    for name in NAMES:
        v = _raw(st, E, name, now)
        json.dumps(v)
    for r in _raw(st, E, "client.stack_events", now):
        assert type(r[0]) is int and type(r[1]) is float and type(r[2]) is float \
            and type(r[3]) is int
    for v in _raw(st, E, "client.stack_set", now).values():
        assert type(v["n"]) is int and type(v["first_ts"]) is float


def test_stack_ids_match_r2_formula():
    st, eng = make_store(), ClientStackEngine()
    now = T0 + 900
    _run(eng, st, now, [_chrome(E, now - 10), _pyreq(E, now - 5)])
    rows = _raw(st, E, "client.stack_events", now)
    want = {S.stack_id(S.stack_token(CHROME_JA3, UA_CHROME_WIN, 120, 64240, None)),
            S.stack_id(S.stack_token(PY_JA3, UA_REQUESTS, 57, 29200, None))}
    assert {r[0] for r in rows} == want


def test_engine_declaration_and_scheduled_run():
    eng = ClientStackEngine()
    assert eng.name == "raw.client_stack" and eng.layer == "raw" and eng.interval == 1
    assert set(eng.produces) == set(NAMES)
    st = make_store()
    now = T0 + 900
    n = run_engine(eng, st, now, observations=[_chrome(E, now - 1)], scheduled=True)
    assert n == 6
    rec = st.health()[eng.name]
    assert rec["ok"] and "unfingerprinted" in rec and "dropped_pseudo" in rec


# ------------------------------------------------------------------ perf
def test_perf():
    rng = random.Random(5)
    uas = [UA_CHROME_WIN, UA_REQUESTS, "okhttp/4.12.0", "curl/8.4.0"]
    ents = [f"10.20.{i // 250}.{i % 250}" for i in range(40)]
    now = T0 + 900
    plain = [obs(SYS, e, now - rng.random() * 900, ja3=CHROME_JA3, user_agent=rng.choice(uas),
                 ttl=rng.choice([57, 120, 250]), win_size=64240, bytes_up=100, bytes_down=900)
             for e in ents for _ in range(50)]                  # 2000 per-event observations
    agg = [obs(SYS, e, now - 900, ja3=CHROME_JA3, user_agent=rng.choice(uas), ttl=120,
               win_size=64240, extra={"count": 80, "bytes_up_total": 8000,
                                      "ts_sample": sorted(rng.random() * 900 for _ in range(64))})
           for e in ents for _ in range(30)]                    # 1200 aggregates x 64 samples
    for batch, bound in ((plain, 0.05), (agg, 0.10)):
        st, eng = make_store(), ClientStackEngine()
        _run(eng, st, now, batch)                                # warm the memo
        best = math.inf
        for k in range(3):
            st = make_store()
            t = time.perf_counter()
            _run(eng, st, now + k, batch)
            best = min(best, time.perf_counter() - t)
        assert best < bound, best
