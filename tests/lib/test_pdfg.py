"""lib/pdfg: action dictionary, tracked sketches, delay bands, mining and
sequence scores (docs/lib3/progressive.md §6.14)."""
from __future__ import annotations

import numpy as np
import pytest

from app.engines.behavior.lib import pdfg as DF

T = 1_790_000_000.0


def test_hash_bloom_and_packing():
    h = [DF.h64(k) for k in ("GET a /x", "GET a /y", "POST a /z")]
    assert len(set(h)) == 3
    bits = 0
    for x in h[:2]:
        bits |= DF.bloom_bits(x)
    assert DF.bloom_has(bits, h[0]) and DF.bloom_has(bits, h[1])
    assert DF.unpack_hashes(DF.pack_hashes(h)) == h


def test_bloom_false_positive_rate_at_ten_actions():
    r = np.random.default_rng(0)
    fp, n = 0, 0
    for trial in range(300):
        bits = 0
        for _ in range(10):
            bits |= DF.bloom_bits(int(r.integers(0, 2 ** 63)))
        for _ in range(20):
            fp += DF.bloom_has(bits, int(r.integers(0, 2 ** 63)))
            n += 1
    assert fp / n < 0.02                   # ~0.5 % expected; the 64-bit single hash gives ~15 %


def test_route_key_fallbacks():
    get = {"tls.sni": "mail.corp.example.com"}.get
    # the service host (integration 2026-09-30: eTLD+1 merged mail. and git. of one domain)
    assert DF.route_key(lambda a: get(a)) == "TLS mail.corp.example.com"
    assert DF.route_key(lambda a: {"tls.sni": "a12.cdn.example.com:443"}.get(a)) == "TLS a{n}.cdn.example.com"
    assert DF.route_key(lambda a: {"net.dst": "10.0.0.5:25"}.get(a)) == "DST 10.0.0.5:25"
    assert DF.route_key(lambda a: {"http.route": "GET h /x", "tls.sni": "a.b"}.get(a)) == "GET h /x"
    assert DF.route_key(lambda a: None) is None
    assert DF.split_key("POST h /x#v12") == ("POST h /x", 12) and DF.split_key("GET h /y") == ("GET h /y", 0)


def test_delay_band_brackets_the_truth():
    r = np.random.default_rng(1)
    h = [0.0] * DF.N_BINS
    for d in r.uniform(60, 240, 500):
        h[DF.delay_bin(d)] += 1
    lo, hi = DF.band(h)
    assert 55 <= lo <= 90 and 180 <= hi <= 260
    assert DF.delay_bin(7200) == DF.N_BINS - 1 and DF.delay_bin(0.3) == 0 and DF.delay_bin(None) is None


def test_tracked_ss_reports_every_eviction():
    """The victim reported by add() is exactly the key that left the sketch,
    under random adds and discards (oracle: set difference)."""
    r = np.random.default_rng(2)
    t = DF.TrackedSS(8)
    for step in range(3000):
        if r.random() < 0.1 and len(t):
            k = t.keys()[int(r.integers(0, len(t)))]
            before = set(t.keys())
            assert t.discard(k)
            assert before - set(t.keys()) == {k}
            continue
        k = int(r.zipf(1.5)) % 50
        before = set(t.keys())
        v = t.add(k, T + step, 1.0, 1.0)
        after = set(t.keys())
        gone = before - after
        assert (v is None and not gone) or gone == {v}


def test_action_ids_are_never_reused():
    ad = DF.ActionDict(k=4)
    seen = {}
    retired = []
    for i in range(40):
        key = f"GET h /p{i % 12}"
        aid, gone = ad.add(key, T + i, 1.0, 1.0)
        if gone is not None:
            retired.append(gone)
        if key in seen and seen[key] != aid:
            assert seen[key] in retired               # a returning key gets a NEW id
        seen[key] = aid
    ids = list(ad.i2k)
    assert retired and ad.retired == len(retired)
    assert not set(ids) & set(retired)
    assert max(retired) < ad.next_id and len(set(retired)) == len(retired)


def test_retire_purges_every_statistic():
    st = DF.FlowState()
    a, _ = st.acts.add("A", T, 1, 1)
    b, _ = st.acts.add("B", T, 1, 1)
    c, _ = st.acts.add("C", T, 1, 1)
    for g in ("*", "7"):
        st.scopes_seen.add(g)
        st.cnt.add((g, b), T, 1, 1)
        st.add_edge(g, a, b, T, 1, 1, 3)
        st.add_edge(g, b, c, T, 1, 1, 3)
        st.add_prec(g, c, a, T, 1, 1)
        st.add_loop(g, a, b, T, 1, 1)
    st.retire(b)
    assert all(b not in k[1:] for k in st.edges.keys())
    assert all(b not in k[1:] for k in st.prec.keys()) and all(b not in k[1:] for k in st.loop2.keys())
    assert all(k[1] != b for k in st.cnt.keys())
    assert not any(b in v for v in st.succ.values()) and ("*", b) not in st.succ
    assert all(b not in k[1:] for k in st.hist)
    assert ("*", c, a) in st.prec                              # unrelated statistics stay


def test_dependency_and_loop_measures():
    assert DF.dependency(40, 0) == pytest.approx(40 / 41)
    assert DF.dependency(37.5, 4.63) < 0.8                    # a loop-y process fails classic dep ...
    assert DF.loop2_measure(4.6, 1.0) >= DF.L2_MIN            # ... the length-two loop measure explains it
    assert DF.loop2_measure(0.0, 0.0) == 0.0


def _chain(st, g, seq, t, gap=60.0, start=True):
    ids = [st.acts.add(k, t, 1, 1)[0] for k in seq]
    for x in ids:
        st.cnt.add((g, x), t, 1, 1)
    if start:
        st.starts.add((g, ids[0]), t, 1, 1)
    for x, y in zip(ids, ids[1:]):
        st.add_edge(g, x, y, t, 1, 1, DF.delay_bin(gap))
    for j, y in enumerate(ids):
        if y in st.top_b or len(st.top_b) < DF.TOP_B:
            st.top_b.add(y)
            st.pcnt.add((g, y), t, 1, 1)
            for x in ids[:j]:
                st.add_prec(g, y, x, t, 1, 1)
    return ids


def test_mine_scope_edges_workflow_and_requires():
    st = DF.FlowState()
    for k in range(20):
        _chain(st, "*", ["login", "home", "list", "approve"], T + 3600 * k)
    sc = DF.mine_scope(st, "*", T + 3600 * 20)
    pairs = {(st.acts.key_of(e["a"]), st.acts.key_of(e["b"])) for e in sc["edges"]}
    assert pairs == {("login", "home"), ("home", "list"), ("list", "approve")}
    assert all(e["dep"] >= 0.8 and 32 <= e["band"][0] <= e["band"][1] <= 64 for e in sc["edges"])   # the 60-s bin
    paths = [[st.acts.key_of(x) for x in w["path"]] for w in sc["workflows"]]
    assert ["login", "home", "list", "approve"] in paths
    req = {(st.acts.key_of(r["b"]), st.acts.key_of(r["a"])) for r in sc["requires"]}
    assert ("approve", "list") in req and ("home", "login") in req


def test_edges_need_ten_units_and_five_percent():
    st = DF.FlowState()
    for k in range(9):
        _chain(st, "*", ["a", "b"], T + 3600 * k)
    assert not DF.mine_scope(st, "*", T + 9 * 3600)["edges"]           # 9 < 10 evidence units
    for k in range(9, 30):
        _chain(st, "*", ["a", "b"], T + 3600 * k)
    _chain(st, "*", ["a", "c"], T + 31 * 3600)
    keys = {st.acts.key_of(e["b"]) for e in DF.mine_scope(st, "*", T + 32 * 3600)["edges"]}
    assert keys == {"b"}


def test_edges_that_stopped_become_stale():
    """A renamed / abandoned step: its confidence-channel evidence lasts for
    weeks, but after a silent week (and >= 3 expected occurrences) the edge is
    no longer kept; a weekend of silence never makes a daily edge stale."""
    st = DF.FlowState()
    for k in range(15):
        _chain(st, "*", ["list", "item"], T + 86400 * k)
    last = T + 86400 * 14
    assert DF.mine_scope(st, "*", last + 3 * 86400)["edges"]            # a long weekend: still kept
    sc = DF.mine_scope(st, "*", last + 8 * 86400)
    assert not sc["edges"] and sc["stale_edges"] == 1
    assert DF.edge_stale(10.0, last, last + 6 * 86400) is False


def test_edge_staleness_counts_normal_days_of_its_day_type():
    """With P01's calendar (§6.8.1): a daily workday edge is stale after 2
    missed normal workdays; a weekend, a holiday week or abnormal days never
    count; a weekly edge needs ~3 weeks of missed normal workdays."""
    D0 = 740000                                            # a Monday (ordinal % 7 == 0)
    cls = {d: (0 if (d - D0) % 7 < 5 else 1) for d in range(D0, D0 + 60)}
    norm = {d: True for d in range(D0, D0 + 60)}
    daily = [D0, D0 + 11, 10, 0]                           # every workday of two weeks (Mon..Fri of week 2)
    assert DF.edge_stale_days(daily, cls, norm, D0 + 14) is False         # Sat, Sun missed: not workdays
    assert DF.edge_stale_days(daily, cls, norm, D0 + 15) is False         # Mon missed (k = 1)
    assert DF.edge_stale_days(daily, cls, norm, D0 + 16) is True          # Mon, Tue missed (k = 2)
    hol = dict(norm)
    hol.update({d: False for d in range(D0 + 14, D0 + 21)})                # a holiday week
    assert DF.edge_stale_days(daily, cls, hol, D0 + 21) is False
    weekly = [D0, D0 + 14, 3, 0]                           # three Mondays in 15 workdays
    assert DF.edge_stale_days(weekly, cls, norm, D0 + 21) is False
    assert DF.edge_stale_days(weekly, cls, norm, D0 + 35) is True
    assert DF.edge_stale_days(daily, cls, {}, D0 + 30) is None            # no calendar: time rule


def test_mine_scope_uses_the_calendar_when_present():
    st = DF.FlowState()
    D0 = 740000
    cls = {d: (0 if (d - D0) % 7 < 5 else 1) for d in range(D0, D0 + 30)}
    st.set_calendar(cls, {}, D0)
    for k in range(15):                                    # every workday of three weeks
        d = D0 + k + 2 * (k // 5)
        ids = _chain(st, "*", ["list", "item"], T + 86400 * k)
        st._edge_day(("*", ids[0], ids[1]), d)
    last_day = D0 + 18                                     # Friday of week 3
    st.set_calendar(cls, {d: True for d in range(D0, last_day + 4)}, last_day + 4)
    assert DF.mine_scope(st, "*", T + 86400 * 16)["edges"]                # weekend + Mon missed
    st.set_calendar(cls, {d: True for d in range(D0, last_day + 5)}, last_day + 5)
    sc = DF.mine_scope(st, "*", T + 86400 * 17)
    assert not sc["edges"] and sc["stale_edges"] == 1                      # Mon, Tue missed
