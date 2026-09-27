"""B10 SequenceEngine (docs/lib3/engines.md '## B10'): spec unit tests (a)-(e)
plus edge cases. Engines only talk to the store, so a tiny R2 stand-in
(`Sys`) writes model.template and act.stream exactly as m_template documents.

Spec test mapping:
  (a) test_a_out_of_grammar_session_excess
  (b) test_b_random_walk_conformal_fpr
  (c) test_c_unseen_source_token_scored
  (d) test_d_role_rename_keeps_class_model
  (e) test_e_stream_frac_weights_counts
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from helpers import DT, T0, make_store, run_engine, set_trust

from app.engines.behavior.lib import combine, emit
from app.engines.behavior.lib import gating as G
from app.engines.behavior.lib import m_seq
from app.engines.behavior.lib import m_template as MT
from app.engines.behavior.lib import ppm as P
from app.engines.behavior.lib.classkeys import SYSTEM_KEY
from app.engines.behavior.sequence import CLASS_LLR, SequenceEngine
from app.models.schema import RawMetric, SignatureMatch

S = "erp"
HOST = "app.corp"
E = "10.0.0.1"
D_TICKS = G.commit_delay_ticks(DT)            # 4 at 900 s


class Sys:
    """R2 stand-in: one Templater per system (model.template) and act.stream /
    act.stream_frac / act.events writers. Tokens are interned once per event,
    as R2 does, so the 'system count < 3 -> {rare}' floor behaves as live."""

    def __init__(self, store, system: str = S, paths=()) -> None:
        self.store, self.s = store, system
        m = MT.new_model()
        self.tpl = m["templater"]
        store.put_model(system, SYSTEM_KEY, MT.MODEL, m)
        for p in paths:
            self.warm(p)

    def warm(self, path: str, method: str = "GET", statuses=(200,)) -> None:
        """An established system: the literal path is settled in the prefix tree
        (> 2 hits) and its tokens have a system count >= 3 (not '{rare}')."""
        for _ in range(3):
            self.tpl.template_path(HOST, method, path)
        for st_ in statuses:
            self.tpl.intern(self.tok(path, st_, method), 5.0)

    def tok(self, path: str, status: int = 200, method: str = "GET") -> str:
        return self.tpl.http_token(method, HOST, path, status, w=0.0)[0]

    def write(self, e: str, now: float, events, frac: float = 1.0, outcome: int = 2) -> None:
        rows = [(ts, self.tpl.intern(t, 1.0), outcome, 100.0, 1000.0, 0, 0) for ts, t in events]
        arr = np.array(rows, dtype=MT.STREAM_DTYPE)
        arr.flags.writeable = False
        for name, v in (("act.stream", arr), ("act.stream_frac", frac),
                        ("act.events", float(len(rows)) / frac)):
            self.store.add_raw(RawMetric(name=name, value=v, ts=now, system=self.s, entity=e))


def session(sy: Sys, t0: float, paths, rng, gap=(4.0, 15.0)):
    ev, t = [], t0
    for p in paths:
        ev.append((t, sy.tok(p)))
        t += float(rng.uniform(*gap))
    return ev


def clerk_paths(rng):
    n = int(rng.integers(1, 5))
    return ["/login", "/dashboard", "/orders"] + [f"/orders/view/{int(rng.integers(1000, 99999))}"
                                                  for _ in range(n)]


def last(store, e=E, system=S):
    return store.profile(system, e).extra["sequence"]["last"]


def score(store, now, d="seq", e=E, system=S):
    return emit.read_row(store, system, e, emit.SCORE, now).get(d, math.nan)


BASE = ["/login", "/dashboard", "/orders", "/orders/view/1", "/orders/export", "/logout"]


def train_clerk(store, sy, eng, n_ticks=150, e=E, seed=1, t0=T0):
    """2 sessions per 15-min tick (300 sessions over 150 ticks), training mode."""
    rng = np.random.default_rng(seed)
    for i in range(n_ticks):
        now = t0 + i * DT
        ev = session(sy, now - 850, clerk_paths(rng), rng) + session(sy, now - 400,
                                                                      clerk_paths(rng), rng)
        sy.write(e, now, ev)
        run_engine(eng, store, now, training=True)
    return t0 + n_ticks * DT


# ------------------------------------------------------------------ spec (a)
@pytest.mark.parametrize("export_known", [True, False])
def test_a_out_of_grammar_session_excess(export_known):
    st = make_store()
    sy = Sys(st, paths=BASE)
    eng = SequenceEngine()
    if export_known:
        # a manager uses /orders/export regularly: known to the system, never to E
        rng = np.random.default_rng(5)
        for i in range(40):
            now = T0 + i * DT
            sy.write("10.0.0.9", now, session(sy, now - 700, ["/login", "/orders/export"], rng))
    now = train_clerk(st, sy, eng)
    m = m_seq.get(st, S, E)
    assert 120.0 <= m["session_gap"] < 400.0          # valley between ~10 s and ~7 min gaps
    mu, _ = m_seq.entropy_rate(m)
    assert 0.0 < mu < 0.5
    # attack: login -> export -> login -> export
    ev = [(now - 800 + 10 * k, sy.tok(p)) for k, p in
          enumerate(["/login", "/orders/export", "/login", "/orders/export"])]
    sy.write(E, now, ev)
    run_engine(eng, st, now)
    atk = last(st)
    assert atk["excess"] >= 4.0
    assert score(st, now) >= 4.0
    # a normal held-out session
    now += DT
    sy.write(E, now, session(sy, now - 800, clerk_paths(np.random.default_rng(99)),
                             np.random.default_rng(98)))
    run_engine(eng, st, now)
    assert last(st)["excess"] < 1.0
    assert score(st, now) < 4.0


# ------------------------------------------------------------------ spec (b)
def test_b_random_walk_conformal_fpr():
    """High-entropy random walk (log2 5 bits/step over 40 pages). Scores are
    relative to the entity's own entropy rate, so a per-entity randomised
    conformal p (B24's rule, rolling ring of 256) is at nominal: FPR <= 1%
    up to binomial noise (2 sd at n = 1000; the realised rate over seeds 1-7
    was 0.9-1.4 %). A v1-style fixed 4-bit surprisal rule fires on most ticks."""
    st = make_store()
    NV = 40
    sy = Sys(st, paths=[f"/w/p{j}" for j in range(NV)])
    toks = [sy.tok(f"/w/p{j}") for j in range(NV)]
    rng = np.random.default_rng(3)
    eng = SequenceEngine()
    cur, scores, fixed_hits = 0, [], 0
    n_ticks = 1300
    for i in range(n_ticks):
        now = T0 + i * DT
        ev, t = [], now - 850
        for _ in range(int(rng.integers(15, 40))):
            cur = (cur + int(rng.integers(-2, 3))) % NV
            ev.append((t, toks[cur]))
            t += float(rng.uniform(3, 15))
        sy.write(E, now, ev)
        run_engine(eng, st, now, training=True)
        scores.append(score(st, now))
        if i >= 300:
            m = m_seq.get(st, S, E)
            bits = m_seq.loglik(m, [s for s in m["_rows"][now].segs[0][1]],
                                m_seq.backoff(st, S, E), m_seq.vocab_size(st, S))
            fixed_hits += bool(np.max(bits) >= 4.0)
    sc = np.asarray(scores)
    assert np.isfinite(sc[300:]).all()
    ring = np.zeros(0)
    fp = n = 0
    for i, s in enumerate(sc):
        if not np.isfinite(s):
            continue
        if i >= 300:
            p = combine.randomized_conformal_p(ring, s, combine.seeded_uniform(S, E, "seq", i))
            n += 1
            fp += p <= 0.01
        ring = np.sort(np.append(ring, np.float32(s)))[-256:] if ring.size < 256 else \
            np.sort(np.append(np.delete(ring, 0), np.float32(s)))
    fpr = fp / n
    assert n == 1000
    assert fpr <= 0.01 + 2.0 * math.sqrt(0.01 * 0.99 / n)
    assert fixed_hits / 1000 > 0.2


# ------------------------------------------------------------------ spec (c)
def test_c_unseen_source_token_scored():
    st = make_store()
    sy = Sys(st, paths=BASE + ["/admin/users"])
    eng = SequenceEngine()
    now = train_clerk(st, sy, eng, n_ticks=40)
    m = m_seq.get(st, S, E)
    smap = m_seq.SymbolMap.from_store(st, S)
    known = sy.tok("/login")
    seen_src = [known, sy.tok("/dashboard")]
    unseen_src = [sy.tok("/admin/users"), sy.tok("/dashboard")]       # source never seen by E
    bk, V = m_seq.backoff(st, S, E), smap.vocab_size
    b_seen = m_seq.loglik(m, seen_src, bk, V)
    b_unseen = m_seq.loglik(m, unseen_src, bk, V)
    assert np.isfinite(b_unseen).all() and (b_unseen > 0).all()
    assert b_unseen[1] > b_seen[1]                   # v1 scored the unseen source 0
    # a never-seen system token (rare symbol) mid-session is scored > 0 as well
    b_rare = m_seq.loglik(m, [known, "{rare:http}", sy.tok("/dashboard")], bk, V)
    assert (b_rare[1:] > 1.0).all()
    # through the engine: finite, positive score
    sy.write(E, now, [(now - 800, sy.tok("/admin/users")), (now - 790, sy.tok("/dashboard"))])
    run_engine(eng, st, now)
    assert score(st, now) > 0.0 and last(st)["excess"] > 0.0


# ------------------------------------------------------------------ spec (d)
def test_d_role_rename_keeps_class_model():
    st = make_store()
    sy = Sys(st, paths=BASE)
    ents = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    cls = {"assign": {f"{S}|{e}": {"role": "r7", "prob": 0.9} for e in ents},
           "roles": {"r7": {"name": "clerk", "members": [f"{S}|{e}" for e in ents]}},
           "version": 1}
    st.put_model("__org__", "__org__", "model.class", cls)
    eng = SequenceEngine()
    rng = np.random.default_rng(2)

    def tick(i):
        now = T0 + i * DT
        for e in ents:
            sy.write(e, now, session(sy, now - 800, clerk_paths(rng), rng))
        run_engine(eng, st, now, training=True)
        return now

    for i in range(12):
        tick(i)
    cm = m_seq.get(st, S, "class:r7")
    assert cm is not None and cm["kind"] == "class" and cm["members"] == 3
    v0 = cm["version"]
    cls["roles"]["r7"]["name"] = "teller"            # rename only
    cls["version"] = 2
    for i in range(12, 24):
        now = tick(i)
    cm2 = m_seq.get(st, S, "class:r7")
    assert cm2["version"] > v0 and cm2["ppm"].n_tokens >= cm["ppm"].n_tokens
    pseudo = st.pseudo_entities(S)
    assert "class:r7" in pseudo and not any("clerk" in p or "teller" in p for p in pseudo)
    assert m_seq.get(st, S, E)["class_key"] == "class:r7"
    assert st.profile(S, E).extra["sequence"]["class_key"] == "class:r7"
    # class tier is a backoff tier and class_llr is written for members
    assert len(m_seq.backoff(st, S, E)) == 2
    t, M = st.vec_since(S, E, CLASS_LLR, now)
    assert t.size == 1 and np.isfinite(M[0, 0])


# ------------------------------------------------------------------ spec (e)
def test_e_stream_frac_weights_counts():
    vocabs = []
    for frac in (1.0, 0.25):
        st = make_store()
        sy = Sys(st, paths=BASE)
        eng = SequenceEngine()
        rng = np.random.default_rng(4)
        sy.write(E, T0, session(sy, T0 - 800, clerk_paths(rng), rng), frac=frac)
        for i in range(D_TICKS + 1):
            run_engine(eng, st, T0 + i * DT, training=True)
        m = m_seq.get(st, S, E)
        assert len(m["_gate"].journal) == 1
        vocabs.append(P.vocab(m["ppm"]))
        g = m_seq.gap_hist(m)
        vocabs.append({"_gap": float(g.sum())})
    full, _, quarter, _ = vocabs
    assert full and set(full) == set(quarter)
    for k in full:
        assert quarter[k] == pytest.approx(4.0 * full[k], rel=1e-12)


# ------------------------------------------------------------------ edges
def test_empty_store_and_silent_entity():
    st = make_store()
    eng = SequenceEngine()
    assert run_engine(eng, st, T0) == 0
    # a known entity zero-filled by R2 (act.events = 0, touch=False): nothing written
    st.add_raw(RawMetric(name="act.events", value=3.0, ts=T0 - DT, system=S, entity=E))
    st.add_raw(RawMetric(name="act.events", value=0.0, ts=T0, system=S, entity=E), touch=False)
    assert run_engine(eng, st, T0) == 0
    assert m_seq.get(st, S, E) is None
    assert st.vec_at(S, E, emit.SCORE, T0) is None
    assert st.events() == []


def test_stale_stream_is_degraded_not_p1():
    st = make_store()
    sy = Sys(st, paths=BASE)
    eng = SequenceEngine()
    now = train_clerk(st, sy, eng, n_ticks=10)
    # R2 counted events but wrote no stream this tick
    st.add_raw(RawMetric(name="act.events", value=7.0, ts=now, system=S, entity=E))
    run_engine(eng, st, now)
    row = emit.read_array(st, S, E, emit.SCORE, now)
    from app.engines.behavior.lib.detectors import DETECTOR_INDEX as DI
    assert math.isnan(row[DI["seq"]]) and math.isnan(row[DI["dwell"]])
    deg = emit.read_dict(st, S, E, emit.DEGRADED, now)
    assert deg["seq"].startswith("stale") and deg["dwell"].startswith("stale")
    # an R2 failure at this tick is a producer error
    now += DT
    st.put_health("raw.action_token", {"last_error_ts": now})
    run_engine(eng, st, now)
    assert emit.read_dict(st, S, E, emit.DEGRADED, now)["seq"].startswith("producer_error")


def test_nan_inputs():
    st = make_store()
    sy = Sys(st, paths=BASE)
    eng = SequenceEngine()
    now = train_clerk(st, sy, eng, n_ticks=20)
    tid = sy.tpl.intern(sy.tok("/login"), 1.0)
    arr = np.array([(math.nan, tid, 2, 0, 0, 0, 0), (now - 500, tid, 2, 0, 0, 0, 0),
                    (math.inf, tid, 2, 0, 0, 0, 0)], dtype=MT.STREAM_DTYPE)
    st.add_raw(RawMetric(name="act.stream", value=arr, ts=now, system=S, entity=E))
    st.add_raw(RawMetric(name="act.stream_frac", value=math.nan, ts=now, system=S, entity=E))
    run_engine(eng, st, now)
    m = m_seq.get(st, S, E)
    row = m["_rows"][now]
    assert row.n_tok == 1 and row.w == 1.0          # NaN / inf ts dropped, NaN frac -> 1
    assert math.isfinite(score(st, now))
    # a tick whose rows are all non-finite: no tokens, no score, no crash
    now += DT
    bad = np.array([(math.nan, tid, 2, 0, 0, 0, 0)], dtype=MT.STREAM_DTYPE)
    st.add_raw(RawMetric(name="act.stream", value=bad, ts=now, system=S, entity=E))
    run_engine(eng, st, now)
    assert now not in m_seq.get(st, S, E)["_rows"]


def test_training_emits_no_events_and_learns():
    st = make_store()
    sy = Sys(st, paths=BASE)
    eng = SequenceEngine()
    now = train_clerk(st, sy, eng, n_ticks=30)
    sy.write(E, now, [(now - 800, sy.tok("/login")), (now - 790, sy.tok("/orders/export"))])
    run_engine(eng, st, now, training=True)
    assert st.events() == []
    m = m_seq.get(st, S, E)
    assert len(m["_gate"].journal) == 30 + 1 - D_TICKS
    # live without trust (fail safe): rows are processed but carry weight 0
    n_before = m["ppm"].n_tokens
    for i in range(1, D_TICKS + 3):
        t = now + i * DT
        sy.write(E, t, [(t - 800, sy.tok("/login"))])
        run_engine(eng, st, t, training=False)
    m = m_seq.get(st, S, E)
    assert m["ppm"].n_tokens <= n_before * 1.0000001
    assert st.events() == []


def test_live_trust_gates_commits_and_quarantine_holds():
    st = make_store()
    sy = Sys(st, paths=BASE)
    eng = SequenceEngine()
    rng = np.random.default_rng(8)
    ts_all = []
    for i in range(12):
        now = T0 + i * DT
        ts_all.append(now)
        sy.write(E, now, session(sy, now - 800, clerk_paths(rng), rng))
        # trust 0.5 for the first 6 ticks, then quarantine
        set_trust(st, S, E, [now], 0.5, prov=1.0, quarantine=1.0 if i >= 6 else 0.0)
        run_engine(eng, st, now)
    m = m_seq.get(st, S, E)
    gate = m["_gate"]
    assert gate.journal and all(r.w_eff == 0.5 for r in gate.journal)
    # quarantine(t-1) is checked when a row falls due (t = row ts + D): from tick 7
    # on, every due row is held
    assert [r.ts for r in gate.held] == ts_all[7 - D_TICKS:12 - D_TICKS]
    # held rows stay fetchable after act.stream retention (1 h) has pruned the stream
    held_ts = [r.ts for r in gate.held]
    assert all(t in m["_rows"] or t in m["_held"] for t in held_ts)
    # release commits them with trust_prov (the row falling due at the release tick
    # still sees quarantine(t-1) = 1 and is held)
    st.put_model(S, E, "model.control", {"release": [held_ts[0], ts_all[-1]]})
    now = ts_all[-1] + DT
    set_trust(st, S, E, [now], 1.0)
    run_engine(eng, st, now)
    gate = m_seq.get(st, S, E)["_gate"]
    committed = {r.ts: r.w_eff for r in gate.journal}
    assert all(committed.get(t) == 1.0 for t in held_ts)
    assert [r.ts for r in gate.held] == [ts_all[12 - D_TICKS]]


def test_rollback_restores_offline_fit():
    """rollback_to tau: the model equals a fit on the rows <= tau (checkpoint +
    replay from B10's own row buffer, act.stream having been pruned)."""
    st = make_store()
    sy = Sys(st, paths=BASE)
    eng = SequenceEngine()
    rng = np.random.default_rng(11)
    snaps = {}
    for i in range(30):
        now = T0 + i * DT
        sy.write(E, now, session(sy, now - 800, clerk_paths(rng), rng))
        run_engine(eng, st, now, training=True)
        m = m_seq.get(st, S, E)
        snaps[m["_gate"].journal[-1].ts if m["_gate"].journal else None] = \
            (dict(P.vocab(m["ppm"])), list(m["ppm"].stats))
    tau = T0 + 17.5 * DT
    # B28: quarantine = 1 at or before the tick where it writes rollback_to
    set_trust(st, S, E, [now], 1.0, quarantine=1.0)
    st.put_model(S, E, "model.control", {"rollback_to": tau})
    run_engine(eng, st, now + DT, training=True)
    m = m_seq.get(st, S, E)
    assert m["_gate"].journal[-1].ts == T0 + 17 * DT
    ref_vocab, ref_stats = snaps[T0 + 17 * DT]
    got = P.vocab(m["ppm"])
    assert set(got) == set(ref_vocab)
    for k in got:
        assert got[k] == pytest.approx(ref_vocab[k], rel=1e-9)
    np.testing.assert_allclose(m["ppm"].stats, ref_stats, rtol=1e-9)
    assert m["_gate"].held and m["_gate"].held[0].ts == T0 + 18 * DT


def test_cadence_switch_900_to_60():
    st = make_store()
    sy = Sys(st, paths=BASE)
    eng = SequenceEngine()
    now = train_clerk(st, sy, eng, n_ticks=40)
    rng = np.random.default_rng(12)
    t = now
    written = []
    for i in range(40):                               # 60-s ticks, one session over 8 ticks
        t += 60.0
        if i % 8 < 4:
            ev = session(sy, t - 55, clerk_paths(rng)[:3], rng, gap=(3.0, 8.0))
            sy.write(E, t, ev)
            written.append(t)
        run_engine(eng, st, t, training=True, dt=60.0)
        if i % 8 < 4:
            assert math.isfinite(score(st, t, e=E))
    m = m_seq.get(st, S, E)
    j = [r.ts for r in m["_gate"].journal]
    assert j == sorted(j)
    frontier = G.commit_frontier(t, 60.0)
    assert all(w in set(j) for w in written if w <= frontier)
    assert not any(w in set(j) for w in written if w > frontier)
    # the session gap is still a learned valley, and minute ticks did not split sessions
    assert 120.0 <= m["session_gap"] <= 7200.0


def test_stream_b_categories_one_tick_lag():
    st = make_store()
    eng = SequenceEngine()
    st.register_entity(S, E)
    cats = ["browse", "api", "browse", "admin"]
    for i, c in enumerate(cats):
        now = T0 + i * DT
        # the signature layer of tick i-1 wrote matches at ts = now - DT
        st.add_match(SignatureMatch(system=S, entity=E, ts=now - DT, signature_id="x",
                                    label="l", category=c, confidence=0.9))
        st.add_match(SignatureMatch(system=S, entity=E, ts=now - DT, signature_id="y",
                                    label="l", category="transfer", confidence=0.3))
        st.add_match(SignatureMatch(system=S, entity=E, ts=now, signature_id="z",
                                    label="l", category="scan", confidence=0.9))
        run_engine(eng, st, now, training=True)
    m = m_seq.get(st, S, E)
    rows = [m["_rows"][T0 + i * DT] for i in range(4)]
    # tick i reads the matches of tick i-1 (its category and the 'scan' written at
    # ts = now of tick i-1); confidence < 0.6 ('transfer') is ignored
    assert [r.cats for r in rows[1:]] == [("cat:api+scan",), ("cat:browse+scan",),
                                           ("cat:admin+scan",)]
    assert rows[0].cats == ("cat:browse",)          # first tick: since now - dt, not now
    for i in range(4, 4 + D_TICKS + 1):
        run_engine(eng, st, T0 + i * DT, training=True)
    voc = P.vocab(m_seq.get(st, S, E)["ppm_cat"])
    assert "cat:api+scan" in voc and not any("transfer" in k for k in voc)


def test_dwell_credential_stuffing_axis():
    st = make_store()
    sy = Sys(st, paths=BASE)
    eng = SequenceEngine()
    sy.warm("/login", "POST", (200, 401))            # system-known (other users log in)
    now = train_clerk(st, sy, eng, n_ticks=60)
    # 12 POST /login in a row, mostly 401
    ev = [(now - 800 + 5 * k, sy.tok("/login", 401 if k % 10 else 200, "POST"))
          for k in range(12)]
    sy.write(E, now, ev)
    run_engine(eng, st, now)
    assert score(st, now, "dwell") > 3.0
    axes = emit.read_dict(st, S, E, emit.AXES, now)
    assert "credential" in axes["dwell"] and "sequence" in axes["dwell"]
    assert "credential" in axes["seq"]
    pm = emit.read_row(st, S, E, emit.PM, now)
    assert 0.0 < pm["dwell"] < 1e-3


def test_class_llr_sign():
    """class_llr = log P_e - log P_class: positive on the entity's own grammar,
    negative when it replays another class member's grammar (T9)."""
    st = make_store()
    paths_b = ["/login", "/reports", "/reports/q/1", "/export"]
    sy = Sys(st, paths=BASE + paths_b)
    ents = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]
    st.put_model("__org__", "__org__", "model.class", {
        "assign": {f"{S}|{e}": {"role": "r1", "prob": 1.0} for e in ents},
        "roles": {"r1": {"name": "office"}}})
    eng = SequenceEngine()
    rng = np.random.default_rng(6)

    def analyst(t0):
        n = int(rng.integers(1, 4))
        return ["/login", "/reports"] + [f"/reports/q/{int(rng.integers(1, 999))}"
                                         for _ in range(n)] + ["/export"]

    for i in range(80):
        now = T0 + i * DT
        for e in ents:
            p = clerk_paths(rng) if e == E else analyst(now)
            sy.write(e, now, session(sy, now - 800, p, rng))
        run_engine(eng, st, now, training=True)
    now = T0 + 80 * DT
    sy.write(E, now, session(sy, now - 800, clerk_paths(rng), rng))
    run_engine(eng, st, now)
    own = float(st.vec_at(S, E, CLASS_LLR, now)[0])
    now += DT
    sy.write(E, now, session(sy, now - 800, analyst(now), rng))
    run_engine(eng, st, now)
    other = float(st.vec_at(S, E, CLASS_LLR, now)[0])
    assert own > 0.1 and other < -3.0


def test_perf_many_entities():
    st = make_store()
    NV = 120
    sy = Sys(st, paths=[f"/m{k}/p{j}" for k in range(10) for j in range(12)])
    toks = [sy.tok(f"/m{k}/p{j}") for k in range(10) for j in range(12)]
    rng = np.random.default_rng(0)
    ents = [f"10.1.0.{i}" for i in range(40)]
    st.put_model("__org__", "__org__", "model.class", {
        "assign": {f"{S}|{e}": {"role": f"r{i % 4}", "prob": 1.0} for i, e in enumerate(ents)},
        "roles": {f"r{k}": {"name": f"role{k}"} for k in range(4)}})
    cur = {e: int(rng.integers(NV)) for e in ents}
    eng = SequenceEngine()
    times = []
    for i in range(12):
        now = T0 + i * DT
        for e in ents:
            ev, t = [], now - 800
            for _ in range(40):
                cur[e] = (cur[e] + int(rng.integers(-2, 3))) % NV
                ev.append((t, toks[cur[e]]))
                t += float(rng.uniform(2, 12))
            sy.write(e, now, ev)
        t0 = time.perf_counter()
        run_engine(eng, st, now, training=True)
        times.append(time.perf_counter() - t0)
    per_tok_us = np.median(times[4:]) / (40 * 40) * 1e6
    assert np.median(times[4:]) < 0.25            # generous: ~25 ms measured for 1600 tokens
    assert per_tok_us < 150.0


def test_without_template_model_ids_are_opaque_symbols():
    """No model.template (R2 not deployed yet): token ids become stable opaque
    symbols 'id:<tid>|<outcome>' and the entity is still modelled and scored."""
    st = make_store()
    eng = SequenceEngine()
    rng = np.random.default_rng(13)
    for i in range(30):
        now = T0 + i * DT
        rows = [(now - 800 + 7 * k, tid, 2, 0, 0, 0, 0)
                for k, tid in enumerate([1, 2, 3] + [4] * int(rng.integers(1, 4)))]
        st.add_raw(RawMetric(name="act.stream", value=np.array(rows, dtype=MT.STREAM_DTYPE),
                             ts=now, system=S, entity=E))
        run_engine(eng, st, now, training=True)
    m = m_seq.get(st, S, E)
    assert set(P.vocab(m["ppm"])) == {"id:1|2xx", "id:2|2xx", "id:3|2xx", "id:4|2xx"}
    assert math.isfinite(score(st, now)) and score(st, now) < 4.0
