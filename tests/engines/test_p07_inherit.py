"""P07: a node too young for its own grammar states its nearest ancestor's
grammar when every value it saw obeys it (pgrammar.inherit_text). Pack O,
seed 1 (2026-09-30 run): the 综合部 login node created on day ~19 held 2
logins and its statement had no username grammar at all."""
from __future__ import annotations

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
GA = ["192.168.1.21", "192.168.1.23", "10.168.7.121"]
SALES = [f"192.168.3.{20 + k}" for k in range(12)]
NAMES = ["amy", "brian", "carol", "david", "emma", "frank", "grace", "henry", "ivan", "julia",
         "kevin", "linda"]


def test_inherit_text_needs_every_seen_shape_to_match():
    ts = PN.TextSummary()
    ts.update("jack", T0, 1.0, 1.0)
    anc = {"kind": "text", "grammar": "[a-z]{3,8}", "confidence": 0.9, "closed": ["x"], "_from": 4}
    rec = PG.inherit_text(anc, ts, T0)
    assert rec["inherited"] == 4 and rec["grammar"] == "[a-z]{3,8}" and "closed" not in rec
    assert rec["confidence"] < 0.9
    ts.update("o'brien", T0, 1.0, 1.0)
    assert PG.inherit_text(anc, ts, T0) is None


def test_young_group_node_states_the_route_grammar():
    store = MetricStore()
    orc = OracleLearner(store, "oa", T0, ["POST /login"], {"GA": GA, "S": SALES}, config=CFG)
    eng = PayloadGrammarEngine()
    for d in range(12):
        T = T0 + d * DAY + 12 * 3600
        bb = EV.BatchBuilder("oa")
        rows = [(ip, u) for ip, u in zip(SALES, NAMES)]
        if d == 11:
            rows += [(GA[0], "jack"), (GA[1], "rose")]
        for i, (ip, u) in enumerate(rows):
            bb.add(T - 3600 + i * 7.0, ip, {"http.route": "POST /login", "body.kv.username": u}, 1.0)
        b = bb.build(T - 43200, T)
        store.add_batch("oa", EV.EVT_BATCH, T, b)
        orc.learn(b)
        eng.safe_run(Context(store=store, now=T + 3600, window_s=60, config=dict(CFG)))
    nodes = MP.get_model(store, "oa", MP.PGRAMMAR)["nodes"][0]
    route = orc.node_for("POST /login")
    ga = orc.node_for("POST /login", "GA")
    own = nodes[route]["attrs"]["body.kv.username"]
    rec = nodes[ga]["attrs"]["body.kv.username"]
    assert rec["inherited"] == route and rec["grammar"] == own["grammar"]
    assert "closed" not in rec
