"""lib/m_seq.py (model.seq accessors, owner B10) and B10 governance paths
that consumers rely on: the session-gap valley (D2, B11, B20), the tokeniser
and loglik B16 uses to score candidates under their own models, the dwell NB,
portrait descriptors (B30), link seeding and the freeze directive."""
from __future__ import annotations

import math

import numpy as np
import pytest

from helpers import DT, T0, make_store, run_engine

from app.engines.behavior.lib import m_seq
from app.engines.behavior.lib import m_template as MT
from app.engines.behavior.lib import ppm as P
from app.engines.behavior.lib.classkeys import SYSTEM_KEY
from app.engines.behavior.sequence import SequenceEngine
from app.models.schema import RawMetric

from test_b10_sequence import BASE, E, HOST, S, Sys, clerk_paths, session, train_clerk


def _hist(gaps):
    return np.bincount(m_seq.gap_bins(gaps), minlength=m_seq.GAP_BINS).astype(float)


# ------------------------------------------------------------ session gap
def test_gap_valley_between_modes_and_clipped():
    rng = np.random.default_rng(0)
    within = np.exp(rng.normal(np.log(8.0), 0.6, 400))           # ~8 s think times
    between = np.exp(rng.normal(np.log(4 * 3600.0), 0.5, 40))   # ~4 h between sessions
    g = m_seq.gap_valley(_hist(np.concatenate([within, between])))
    assert 120.0 <= g <= 7200.0
    assert within.max() < g < between.min() or g in (120.0, 7200.0)
    # modes at 10 s and 20 min: the valley lies between them
    b2 = np.exp(rng.normal(np.log(1200.0), 0.3, 60))
    g2 = m_seq.gap_valley(_hist(np.concatenate([within, b2])))
    assert np.quantile(within, 0.99) < g2 < np.quantile(b2, 0.05)
    # clipping: both modes below 2 min
    g3 = m_seq.gap_valley(_hist(np.concatenate([within, np.full(50, 60.0)])))
    assert g3 == 120.0
    # too little data / all mass in one bin / NaN -> default
    assert m_seq.gap_valley(_hist(within[:10])) == m_seq.DEFAULT_SESSION_GAP_S
    assert m_seq.gap_valley(_hist(np.full(100, 5.0))) == m_seq.DEFAULT_SESSION_GAP_S
    h = _hist(np.concatenate([within, between]))
    h[3] = np.nan
    assert 120.0 <= m_seq.gap_valley(h) <= 7200.0


def test_gap_bins_edges():
    b = m_seq.gap_bins([0.0, 1e-9, 0.1, 1.0, 1e9, np.nan])
    assert b[0] == 0 and b[1] == 0 and b[4] == m_seq.GAP_BINS - 1 and b[5] == 0
    assert (b >= 0).all() and (b < m_seq.GAP_BINS).all()


def test_session_starts():
    ts = np.array([0.0, 5.0, 400.0, 405.0, 2000.0])
    cut, first = m_seq.session_starts(ts, 120.0, last_ts=-10.0)
    assert cut.tolist() == [2, 4] and first is False
    assert m_seq.session_starts(ts, 120.0)[1] is True                 # no previous event
    assert m_seq.session_starts(ts, 120.0, last_ts=-500.0)[1] is True


def test_session_gap_is_learned_and_published():
    st = make_store()
    sy = Sys(st, paths=BASE)
    eng = SequenceEngine()
    train_clerk(st, sy, eng, n_ticks=60)
    m = m_seq.get(st, S, E)
    assert m_seq.session_gap(m) == m["session_gap"]
    assert 120.0 <= m["session_gap"] < 400.0
    assert m_seq.session_gap(None) == m_seq.DEFAULT_SESSION_GAP_S
    assert m_seq.session_gap({"session_gap": math.nan}) == m_seq.DEFAULT_SESSION_GAP_S
    sysm = m_seq.get(st, S, SYSTEM_KEY)
    assert sysm["kind"] == "system" and sysm["ppm"].order == 0
    assert 120.0 <= m_seq.session_gap(sysm) <= 7200.0


# ------------------------------------------------------------------ dwell
def test_dwell_sf_prior_and_learned():
    empty = {"dwell": {"h": 1.0, "t_ref": 0.0, "g": 0.0, "fam": {}}}
    # prior: geometric with mean x = 1 -> P(L >= l) = 2^-(l-1)
    assert m_seq.dwell_sf(empty, "f", 1) == 1.0
    assert m_seq.dwell_sf(empty, "f", 4) == pytest.approx(2.0 ** -3, rel=1e-9)
    # a family always seen in runs of exactly 1: a run of 3 is very unlikely
    one = {"dwell": {"h": 1.0, "t_ref": 0.0, "g": 0.0, "fam": {"f": [500.0, 0.0, 0.0]}}}
    assert m_seq.dwell_sf(one, "f", 3) < 1e-3
    # monotone in length, never NaN for finite moments
    long_ = {"dwell": {"h": 1.0, "t_ref": 0.0, "g": 0.0,
                       "fam": {"f": [100.0, 400.0, 2400.0]}}}                # mean x 4, var 8
    ps = [m_seq.dwell_sf(long_, "f", L) for L in range(1, 30)]
    assert all(a >= b for a, b in zip(ps, ps[1:])) and all(0.0 <= p <= 1.0 for p in ps)
    assert m_seq.dwell_sf(long_, "f", 5) > 0.3
    # tier backoff: an unseen family borrows the class's runs (capped weight)
    assert m_seq.dwell_sf(empty, "f", 3, tier=one) < m_seq.dwell_sf(empty, "f", 3)


# -------------------------------------------------------------- tokeniser
def test_symbol_map_rare_floor_outcome_and_family():
    st = make_store()
    m = MT.new_model()
    tpl = m["templater"]
    st.put_model(S, SYSTEM_KEY, MT.MODEL, m)
    for _ in range(3):                                # settle the literal path
        tpl.template_path(HOST, "GET", "/login")
    http = tpl.http_token("GET", HOST, "/login", 200, w=0.0)[0]
    tid_http = tpl.intern(http, 2.0)                  # system count 2: rare
    dns = "A www.example.com"
    tid_dns = tpl.intern(dns, 3.0)
    smap = m_seq.SymbolMap.from_store(st, S)
    assert smap.symbol(tid_http, 2) == "{rare:http}"
    assert smap.symbol(tid_dns, 4) == "A www.example.com|4xx"      # non-HTTP gets outcome
    assert smap.symbol(0, 2) == "{rare:rare}"                       # evicted / unknown id
    assert smap.vocab_size == 2 + m_seq.N_RARE_SYMBOLS
    tpl.intern(http, 1.0)                                           # third hit: known
    smap = m_seq.SymbolMap.from_store(st, S)
    sym, fam, auth = smap.info(tid_http, 4)
    assert sym == http and fam == "http|read|app.corp|login" and auth
    # no template model: opaque but stable symbols
    assert m_seq.SymbolMap(None).symbol(7, 4) == "id:7|4xx"


def test_stream_symbols_sorted_and_finite():
    st = make_store()
    sy = Sys(st, paths=BASE)
    a, b = sy.tok("/login"), sy.tok("/dashboard")
    rows = [(T0 - 5, sy.tpl.intern(b), 2, 0, 0, 0, 0), (T0 - 9, sy.tpl.intern(a), 2, 0, 0, 0, 0),
            (math.nan, sy.tpl.intern(a), 2, 0, 0, 0, 0)]
    st.add_raw(RawMetric(name="act.stream", value=np.array(rows, dtype=MT.STREAM_DTYPE),
                         ts=T0, system=S, entity=E))
    t, syms, fams, auth = m_seq.stream_symbols(st, S, E, T0)
    assert t.tolist() == [T0 - 9, T0 - 5] and syms == [a, b] and auth == [True, False]
    assert m_seq.stream_symbols(st, S, E, T0 + DT)[1] == []        # other tick: empty


def test_is_auth():
    for f in ("http|write|sso.corp|login", "http|read|app|oauth2", "l4|tcp||ssh",
              "http|write|app|api-token", "dns|addr|corp|kerberos"):
        assert m_seq.is_auth(f), f
    for f in ("http|read|app|orders", "http|read|blog|catalogue", "tls|conn|x.com|443", ""):
        assert not m_seq.is_auth(f), f


# ------------------------------------------------ loglik / B16 / portrait
def test_loglik_accessor_matches_engine_and_candidates():
    st = make_store()
    sy = Sys(st, paths=BASE)
    eng = SequenceEngine()
    rng = np.random.default_rng(3)
    for i in range(40):
        now = T0 + i * DT
        sy.write(E, now, session(sy, now - 800, clerk_paths(rng), rng))
        sy.write("10.0.0.2", now, session(sy, now - 800, ["/login", "/orders/export"] * 2, rng))
        run_engine(eng, st, now, training=True)
    me, other = m_seq.get(st, S, E), m_seq.get(st, S, "10.0.0.2")
    V = m_seq.vocab_size(st, S)
    toks = [sy.tok(p) for p in ["/login", "/dashboard", "/orders", "/orders/view/1"]]
    ll_me = m_seq.loglik(me, toks, m_seq.backoff(st, S, E), V)
    ll_other = m_seq.loglik(other, toks, m_seq.backoff(st, S, "10.0.0.2"), V)
    assert ll_me.sum() < ll_other.sum() - 10.0       # the clerk's own model explains it
    # history=None is a session start: identical to an explicit '{bos}' history
    np.testing.assert_array_equal(ll_me, m_seq.loglik(me["ppm"], toks,
                                                      m_seq.backoff(st, S, E), V,
                                                      history=(m_seq.BOS,)))
    # a missing model scores through the tiers only; nothing at all -> uniform code
    assert np.isfinite(m_seq.loglik(None, toks, m_seq.backoff(st, S, E), V)).all()
    assert m_seq.loglik(None, toks, (), 16).tolist() == [4.0] * 4
    # entropy rate / maturity / excess helpers
    mu, sd = m_seq.entropy_rate(me)
    assert 0.0 < mu < 1.0 and sd >= 0.0 and 0.9 < m_seq.maturity(me) <= 1.0
    assert m_seq.session_excess([mu] * 5, mu) == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(m_seq.session_excess([1.0], math.nan))
    assert m_seq.session_excess([mu + 10.0] * 5, mu) == pytest.approx(5.0)


def test_describe_for_portraits():
    st = make_store()
    sy = Sys(st, paths=BASE)
    eng = SequenceEngine()
    train_clerk(st, sy, eng, n_ticks=30)
    d = m_seq.describe(m_seq.get(st, S, E), k=3)
    assert d["top_bigrams"][0]["ngram"] == [sy.tok("/orders/view/1")] * 2
    assert all(m_seq.BOS not in g["ngram"] for g in d["top_bigrams"])
    assert len(d["top_unigrams"]) == 3 and d["vocab"] >= 4
    assert d["long_dwell"][0]["family"] == "http|read|app.corp|orders"
    prof = st.profile(S, E).extra["sequence"]
    assert prof["top_bigrams"] and prof["session_gap_s"] == m_seq.get(st, S, E)["session_gap"]
    assert m_seq.describe(None)["n_tokens"] == 0.0


# ------------------------------------------------------------- governance
def test_link_seeding_adds_half_of_the_source():
    st = make_store()
    sy = Sys(st, paths=BASE)
    eng = SequenceEngine()
    rng = np.random.default_rng(4)
    A, B = "10.0.0.7", "10.0.0.8"
    for i in range(18):
        now = T0 + i * DT
        if i < 12:                                    # then let every pending row commit
            sy.write(A, now, session(sy, now - 800, clerk_paths(rng), rng))
            sy.write(B, now, session(sy, now - 800, ["/login", "/orders/export"], rng))
        run_engine(eng, st, now, training=True)
    va = P.vocab(m_seq.get(st, S, A)["ppm"])
    vb = P.vocab(m_seq.get(st, S, B)["ppm"])
    st.put_model(S, SYSTEM_KEY, "model.link",
                 {"version": 1, "links": [{"from": A, "to": B, "ts": now}]})
    now += DT
    run_engine(eng, st, now, training=True)
    got = P.vocab(m_seq.get(st, S, B)["ppm"])
    for k in va:
        assert got[k] == pytest.approx(vb.get(k, 0.0) + 0.5 * va[k], rel=1e-6)
    # once per link version
    run_engine(eng, st, now + DT, training=True)
    again = P.vocab(m_seq.get(st, S, B)["ppm"])
    assert again[sy.tok("/dashboard")] == pytest.approx(got[sy.tok("/dashboard")], rel=1e-9)


def test_frozen_stops_commits():
    st = make_store()
    sy = Sys(st, paths=BASE)
    eng = SequenceEngine()
    now = train_clerk(st, sy, eng, n_ticks=10)
    n0 = len(m_seq.get(st, S, E)["_gate"].journal)
    st.put_model(S, E, "model.control", {"frozen": True})
    rng = np.random.default_rng(5)
    for i in range(8):
        t = now + i * DT
        sy.write(E, t, session(sy, t - 800, clerk_paths(rng), rng))
        run_engine(eng, st, t, training=True)
    g = m_seq.get(st, S, E)["_gate"]
    assert g.frozen and len(g.journal) == n0 and not g.held
