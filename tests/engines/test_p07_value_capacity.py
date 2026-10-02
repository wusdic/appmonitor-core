"""P07: adaptive capacity of the exact-value sketch (lib/pgrammar.adapt_values).

Pack O (round-2 runs, seeds 0-1, day 14): 销售部's 20 usernames never formed a
closed set because a TextSummary tracks TEXT_VALUES_K = 16 exact values; with
20 equally used values the sketch keeps evicting, its Good-Turing unseen mass
stays ~0.2 for ever and the requirement's "username 取值集合封闭" could not be
stated for any department larger than 16 users. The capacity now grows when
the sketch's own evidence shows a finite population that outgrew it
(coverage-based richness k / (1 - eviction rate), values repeating), up to
VALUES_K_MAX; random tokens and value spaces too large for the bound keep the
base capacity, and the closure is judged on the arrivals since the growth."""
from __future__ import annotations

import numpy as np
from pcontent_oracle import OracleLearner

from app.core.engine import Context
from app.core.store import MetricStore
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import pgrammar as PG
from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.payload_grammar import PayloadGrammarEngine

CFG = {"progressive": {"enabled": True}, "grain_mode": "tick", "strict": True}
DAY = 86400.0
T0 = 1_788_220_800.0
NAMES20 = ["amy", "brandon", "brian", "carol", "david", "emma", "frank", "grace", "henry", "ivy",
           "james", "jonathan", "kevin", "linda", "michelle", "nancy", "oscar", "patricia",
           "quinn", "ryan"]
SALES = [f"192.168.3.{20 + k}" for k in range(20)]


def _feed(ts, values_per_day, days, rng=None):
    """Daily arrivals (9:00 + i x 7 s); P07's hourly fit is simulated once a day."""
    caps = []
    vcap = None
    for d in range(days):
        vals = values_per_day(d)
        for i, v in enumerate(vals):
            ts.update(v, T0 + d * DAY + 9 * 3600 + i * 7.0, 1.0, 1.0)
        t = T0 + d * DAY + 18 * 3600
        vcap = PG.adapt_values(ts, t, vcap)
        caps.append(ts.values.k)
    return vcap, caps, t


def test_twenty_equally_used_values_grow_the_sketch_and_close():
    ts = PN.TextSummary()
    rng = np.random.default_rng(0)
    vcap, caps, t = _feed(ts, lambda d: list(rng.permutation(NAMES20)), 10)
    assert caps[0] == PN.TEXT_VALUES_K and max(caps) == 32       # grown once, bounded
    rec = PG.fit_text(ts, t, vcap=vcap)
    assert rec["closed"] == sorted(NAMES20)
    assert rec["U"] <= PG.CLOSED_U and rec["capacity"] == 32
    # the same arrivals without the growth: never closed (the round-2 behaviour)
    ts0 = PN.TextSummary()
    rng = np.random.default_rng(0)
    for d in range(10):
        for i, v in enumerate(rng.permutation(NAMES20)):
            ts0.update(v, T0 + d * DAY + 9 * 3600 + i * 7.0, 1.0, 1.0)
    rec0 = PG.fit_text(ts0, t)
    assert "closed" not in rec0 and rec0["U"] > 0.1


def test_random_tokens_never_grow():
    ts = PN.TextSummary()
    n = [0]

    def toks(d):
        out = []
        for _ in range(30):
            n[0] += 1
            out.append("tok%08x" % (n[0] * 2654435761 % (1 << 32)))
        return out
    vcap, caps, t = _feed(ts, toks, 10)
    assert set(caps) == {PN.TEXT_VALUES_K}
    assert vcap == {"k": PN.TEXT_VALUES_K, "seg": None, "open": True}   # nothing more is remembered
    assert "closed" not in PG.fit_text(ts, t, vcap=vcap)


def test_cyclic_use_that_thrashes_the_sketch_still_grows_it():
    # each user once a morning in the SAME order: Space-Saving evicts on every
    # arrival (like LRU), its counters look like random tokens'
    ts = PN.TextSummary()
    vcap, caps, t = _feed(ts, lambda d: list(NAMES20), 8)
    assert max(caps) == 32
    assert PG.fit_text(ts, t, vcap=vcap)["closed"] == sorted(NAMES20)


def test_a_population_beyond_the_bound_keeps_the_base_capacity():
    ts = PN.TextSummary()
    rng = np.random.default_rng(1)
    names = [f"user{k:03d}" for k in range(100)]
    vcap, caps, t = _feed(ts, lambda d: list(rng.permutation(names)), 6)
    assert set(caps) == {PN.TEXT_VALUES_K} and vcap.get("seg") is None
    assert len(vcap.get("seen") or ()) <= PG.SEEN_MAX


def test_a_small_set_is_unchanged():
    ts = PN.TextSummary()
    vcap, caps, t = _feed(ts, lambda d: ["jack", "rose", "mike"], 10)
    # (a sketch that is not full remembers nothing)
    assert set(caps) == {PN.TEXT_VALUES_K} and vcap is None
    rec = PG.fit_text(ts, t, vcap=vcap)
    assert rec["closed"] == ["jack", "mike", "rose"] and "capacity" not in rec


def test_a_confidence_reset_restarts_the_segment():
    ts = PN.TextSummary()
    rng = np.random.default_rng(2)
    vcap, caps, t = _feed(ts, lambda d: list(rng.permutation(NAMES20)), 6)
    assert vcap["seg"] is not None and ts.values.k == 32
    ts.reset_confidence(t + 60.0)              # P04 accepted a change at the node
    cap2 = PG.adapt_values(ts, t + 120.0, vcap)
    assert cap2["seg"]["t0"] == t + 120.0 and cap2["k"] == 32  # a new segment, same capacity
    rec = PG.fit_text(ts, t + 120.0, vcap=cap2)
    assert "closed" not in rec                                   # nothing seen since: not closed yet


def test_p07_engine_states_the_twenty_value_closed_set():
    store = MetricStore()
    orc = OracleLearner(store, "oa", T0, ["POST /login"], {"S": SALES}, config=CFG)
    eng = PayloadGrammarEngine()
    for d in range(10):
        T = T0 + d * DAY + 12 * 3600
        bb = EV.BatchBuilder("oa")
        for i, (ip, u) in enumerate(zip(SALES, NAMES20)):
            bb.add(T - 3600 + i * 7.0, ip, {"http.route": "POST /login", "body.kv.username": u}, 1.0)
        b = bb.build(T - 43200, T)
        store.add_batch("oa", EV.EVT_BATCH, T, b)
        orc.learn(b)
        eng.safe_run(Context(store=store, now=T + 3600, window_s=60, config=dict(CFG)))
    nodes = MP.get_model(store, "oa", MP.PGRAMMAR)["nodes"][0]
    rec = nodes[orc.node_for("POST /login")]["attrs"]["body.kv.username"]
    assert rec.get("closed") == sorted(NAMES20), rec.get("U")
    assert rec["capacity"] == 32


def test_a_damped_value_never_joins_the_grown_closed_set():
    """Pack O seed 0, day 21: A3's injected username (one login, damped 0.1 by
    P03) found a free slot in the grown sketch and was listed in 销售部's closed
    set ("admin' OR '1'='1", amy, brandon, ...)."""
    ts = PN.TextSummary()
    rng = np.random.default_rng(3)
    vcap, caps, t = _feed(ts, lambda d: list(rng.permutation(NAMES20)), 8)
    ts.update("admin' OR '1'='1", t + 60.0, 0.1, 0.1)          # learned with outlier damping
    rec = PG.fit_text(ts, t + 120.0, vcap=PG.adapt_values(ts, t + 120.0, vcap))
    assert rec["closed"] == sorted(NAMES20)


def test_a_value_outside_the_grammar_is_never_a_closed_set_member():
    """Pack O seeds 0-1, day 21: A3's injected username, released by B28 after
    its incident and learned as content at full weight, was listed in the
    closed set of a statement whose grammar was [a-z]{3,8}."""
    ts = PN.TextSummary()
    rng = np.random.default_rng(4)
    vcap, caps, t = _feed(ts, lambda d: list(rng.permutation(NAMES20)), 8)
    ts.update("admin' OR '1'='1", t + 60.0, 1.0, 1.0)            # a released row: full weight
    rec = PG.fit_text(ts, t + 120.0, vcap=PG.adapt_values(ts, t + 120.0, vcap))
    assert "'" not in rec["grammar"] and rec["closed"] == sorted(NAMES20)


def test_a_young_child_takes_its_ancestors_capacity_at_once():
    """销售部's login node, split off the department-wide login node on day ~10
    (seed 0), started with the base capacity and needed another day of
    checkpoints: its closed set came on day 21 instead of before day 14."""
    ts = PN.TextSummary()
    rng = np.random.default_rng(5)
    for i, v in enumerate(rng.permutation(NAMES20)):         # one morning of the child
        ts.update(v, T0 + 9 * 3600 + i * 7.0, 1.0, 1.0)
    t = T0 + 18 * 3600
    st0 = PG.adapt_values(ts, t, None)                       # on its own: a first checkpoint only
    assert ts.values.k == PN.TEXT_VALUES_K and st0["seg"] is None
    st = PG.adapt_values(ts, t, st0, inherit_k=64)
    assert ts.values.k == 64 and st["seg"]["k0"] == PN.TEXT_VALUES_K
    for d in (1, 2):
        for i, v in enumerate(rng.permutation(NAMES20)):
            ts.update(v, T0 + d * DAY + 9 * 3600 + i * 7.0, 1.0, 1.0)
    t2 = T0 + 2 * DAY + 18 * 3600
    rec = PG.fit_text(ts, t2, vcap=PG.adapt_values(ts, t2, st, inherit_k=64))
    assert rec["closed"] == sorted(NAMES20)
    # a child that is not full (its own population fits) keeps the base capacity
    small = PN.TextSummary()
    for v in ("jack", "rose", "mike"):
        small.update(v, T0, 1.0, 1.0)
    assert PG.adapt_values(small, T0 + 60, None, inherit_k=64) is None and small.values.k == PN.TEXT_VALUES_K
