"""P07 closure (round 4, lib/pgrammar._closed).

Pack O (round 3, 5 seeds, day 14): finance's three users (16-19 logins on
their node) and the approver's 3-value opinion set (U 0.021) were never stated
closed by PG1's day 14 because closure needed U <= CLOSED_U = 0.02 and
n_c >= CLOSED_N = 20 - an honest Good-Turing U (N1 + E + 0.5) / (N + 1)
cannot be below 0.02 before ~24 arrivals, however clearly the set repeats.
A set is now closed when its unseen class is predicted rarer than its rarest
member (U < e_min / N): every member repeated, nothing evicted. A growing set
(each new value a singleton) and an evicting sketch never close; the stated U
is unchanged."""
from __future__ import annotations

import re

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
FIN = {"192.168.2.10": "lucy", "192.168.2.11": "tom", "192.168.2.12": "kate"}


def _run_engine(days, per_day):
    """per_day(d) -> [(ip, username)]; P07 runs once after each day."""
    store = MetricStore()
    orc = OracleLearner(store, "fin", T0, ["POST /fin/login"], config=CFG)
    eng = PayloadGrammarEngine()
    out = []
    for d in range(days):
        T = T0 + d * DAY + 12 * 3600
        bb = EV.BatchBuilder("fin")
        for i, (ip, u) in enumerate(per_day(d)):
            bb.add(T - 3 * 3600 + i * 60.0, ip, {"http.route": "POST /fin/login", "body.kv.username": u}, 1.0)
        b = bb.build(T - 43200, T)
        store.add_batch("fin", EV.EVT_BATCH, T, b)
        orc.learn(b)
        eng.safe_run(Context(store=store, now=T + 3600, window_s=60, config=dict(CFG)))
        nodes = MP.get_model(store, "fin", MP.PGRAMMAR)["nodes"][0]
        out.append((nodes.get(orc.node_for("POST /fin/login")) or {}).get("attrs", {}).get("body.kv.username"))
    return out


def test_three_finance_users_close_before_twenty_logins():
    """Three users, one login each per workday: closed once each repeated
    (round 3 needed n_c >= 20 and U <= 0.02), with the honest U stated."""
    recs = _run_engine(6, lambda d: list(FIN.items()))
    last = recs[-1]
    assert last["closed"] == ["kate", "lucy", "tom"], (last.get("p_min"), last.get("U"))
    assert last["n_values"] < PG.CLOSED_N and last["U"] > PG.CLOSED_U      # the round-3 rule: open
    first = next(i for i, r in enumerate(recs) if r and r.get("closed"))
    assert first <= 2
    assert recs[0] is None or "closed" not in recs[0]                      # one login each: singletons


def test_opinions_with_rare_values_close_once_each_repeated():
    """The approver's opinions (0.8 / 0.1 / 0.1): closed once both rare values
    recurred, U ~ 0.02 stated (round 3: U 0.021 > 0.02 at n 20-24 - open)."""
    ts = PN.TextSummary()
    seq = ["同意"] * 8 + ["退回"] + ["同意"] * 6 + ["同意，请尽快办理"] + ["同意"] * 4 + ["退回", "同意，请尽快办理"]
    for i, v in enumerate(seq[:-2]):
        ts.update(v, T0 + i * 3 * 3600.0, 1.0, 1.0)
    r = PG.fit_text(ts, T0 + len(seq) * 3 * 3600.0)
    assert "closed" not in r                                    # each rare value seen once
    for i, v in enumerate(seq[-2:]):
        ts.update(v, T0 + (len(seq) - 2 + i) * 3 * 3600.0, 1.0, 1.0)
    r = PG.fit_text(ts, T0 + (len(seq) + 1) * 3 * 3600.0)
    assert r["closed"] == sorted(["同意", "退回", "同意，请尽快办理"]) and r["U"] > PG.CLOSED_U


def test_a_growing_set_never_closes():
    """Three regulars plus one NEW user every day (a department still being
    onboarded): every new value is a singleton, the set stays open."""
    recs = _run_engine(10, lambda d: list(FIN.items()) + [(f"192.168.2.{30 + d}", f"new{d:02d}")])
    assert all(r is None or "closed" not in r for r in recs), [r.get("U") for r in recs if r]


def test_an_evicting_sketch_never_closes():
    """100 users through a 16-value sketch: evictions keep U near 1."""
    ts = PN.TextSummary()
    rng = np.random.default_rng(0)
    names = [f"user{k:03d}" for k in range(100)]
    for d in range(12):
        t = T0 + d * DAY + 12 * 3600
        for i, v in enumerate(rng.permutation(names)[:40]):
            ts.update(str(v), t - 3600 + i * 30.0, 1.0, 1.0)
    rec = PG.fit_text(ts, t)
    assert "closed" not in rec and rec["U"] > 0.5


def test_a_sixty_user_department_grows_and_closes():
    """Pack O's DEV pool: 60 personas, one login each per day. With a 64-value
    bound (round 3) the sketch could not hold them with the 1.25 margin and
    kept evicting (U ~ 0.9, 5 of 5 seeds); the bound now covers <= 102 values."""
    ts = PN.TextSummary()
    rng = np.random.default_rng(6)
    names = [f"dev{k:02d}" for k in range(60)]
    vcap = None
    for d in range(12):                                        # (grows on day 8)
        t = T0 + d * DAY + 12 * 3600
        for i, v in enumerate(rng.permutation(names)):
            ts.update(str(v), t - 3600 + i * 20.0, 1.0, 1.0)
        vcap = PG.adapt_values(ts, t, vcap)
    rec = PG.fit_text(ts, t, vcap=vcap)
    assert ts.values.k == 128 and rec.get("closed") == sorted(names), (ts.values.k, rec.get("U"))


def test_raw_text_counted_as_its_own_shape_is_shaped():
    """A 'shape'-policy text summary fed plain strings (pnode.TextSummary.update
    then counts the raw value as its shape; seen on pack O's replayed OA comment
    rows, seed 1): P07 must read such keys as their phier shape - the comment
    grammar listed the CJK characters of the tracked comments as its charset
    and held 0.7-2 % of fresh comments."""
    rng = np.random.default_rng(2)
    zh = list("的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处府研队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严")
    ts = PN.TextSummary("shape")
    t = T0
    for d in range(10):
        for i in range(12):
            t = T0 + d * DAY + 36000 + i * 600.0
            ts.update("".join(rng.choice(zh, int(rng.integers(10, 60)))), t, 1.0, 1.0)   # a plain str
    rec = PG.fit_text(ts, t)
    rx = re.compile(rec["grammar"])
    fresh = ["".join(rng.choice(zh, int(rng.integers(10, 60)))) for _ in range(50)]
    assert sum(bool(rx.fullmatch(v)) for v in fresh) >= 45, rec["grammar"][:60]
