"""P08: hierarchical pooling of binding evidence (lib/pfd.history_support,
binding._seed_histories).

Pack O (round 2, seed 0, finance): 财务部's three users log in once a workday.
The (net.src -> username) pair was screened on day 8 (three probe rows per
source), the login node's pair sketch started on day 9, and on day 10 P12
switched P08 off because no source had reached n_bind = 5 at the node (gain
0) - the bindings were never stated (PG1 bindings 2/3 on every seed). The
evidence existed: the probe had kept the logins since day 3. Now a screened
pair's tree-level value history starts from the probe's rows, and each source
is judged on its tree-level clean normal DAYS when those carry more evidence
than the node's young sketch and name the same value (the leave-one-out
empirical-Bayes prior over the node's other sources as before)."""
from __future__ import annotations

from pcontent_oracle import OracleLearner

from app.core.engine import Context
from app.core.store import MetricStore
from app.engines.behavior.binding import STATE, BindingEngine
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import pfd as FD
from app.engines.behavior.lib import pnode as PN

CFG = {"progressive": {"enabled": True}, "grain_mode": "tick", "strict": True}
DAY = 86400.0
T0 = 1_788_220_800.0          # a Monday 00:00 UTC
FIN = [("192.168.2.10", "lucy"), ("192.168.2.11", "tom"), ("192.168.2.12", "kate")]
PK = "net.src->body.kv.username"


def _ctx(store, now):
    return Context(store=store, now=now, window_s=60, config=dict(CFG))


def _day(store, orc, eng, rows, d, pages=40):
    T = T0 + d * DAY + 12 * 3600
    bb = EV.BatchBuilder("finance")
    i = 0
    for ip, u in rows:
        bb.add(T - 3600 + i * 7.0, ip, {"http.route": "POST /fin/login", "body.kv.username": u}, 1.0)
        i += 1
    for k in range(pages):                      # the busy rest of the system
        ip = FIN[k % 3][0]
        bb.add(T - 3000 + k * 30.0, ip, {"http.route": "GET /fin/ledger"}, 1.0)
    b = bb.build(T - 43200, T)
    store.add_batch("finance", EV.EVT_BATCH, T, b)
    orc.learn(b)
    for h in range(1, 4):                        # P08 fits hourly
        eng.safe_run(_ctx(store, T + h * 3600))


def _entries(store, orc):
    nid = orc.node_for("POST /fin/login")
    pairs = ((MP.get_model(store, "finance", MP.PBIND)["nodes"].get(0) or {}).get(nid) or {}).get("pairs") or {}
    return (pairs.get(PK) or {}).get("table") or {}


def test_once_a_day_users_are_bound_from_their_days_not_the_young_sketch():
    store = MetricStore()
    orc = OracleLearner(store, "finance", T0, ["POST /fin/login", "GET /fin/ledger"], config=CFG)
    eng = BindingEngine()
    bound_day = None
    for d in range(8):
        _day(store, orc, eng, FIN, d)
        tab = _entries(store, orc)
        if bound_day is None and tab and all((tab.get(ip) or {}).get("bound") for ip, _ in FIN):
            bound_day = d
    st = MP.get_model(store, "finance", STATE)
    h = st["hist"][PK]
    # the history starts with the probe's rows of the days before the sketch
    assert len(h.get("192.168.2.10")["v"]["lucy"][3]) == FD.HIST_DAYS
    # judged on 5 days, two days after the sketch started (screened after 3 days)
    assert bound_day is not None and bound_day <= 4, bound_day
    e = _entries(store, orc)["192.168.2.10"]
    assert e["top"] == "lucy" and e.get("pooled") and e["LB"] >= FD.LB_BIND


def test_a_burst_of_one_day_is_one_day_of_evidence():
    ps = PN.PairSketch()
    h = FD.ValueHistory()
    for d in range(6):
        for x, y in FIN[1:]:
            ps.update(x, y, T0 + d * DAY, 1.0, 1.0)
            h.observe(x, y, T0 + d * DAY, 700000 + d, True)
    for k in range(8):                         # 8 logins of .10 in one session, day 5
        ps.update("192.168.2.10", "lucy", T0 + 5 * DAY + k * 30.0, 1.0, 1.0 / (k + 1))
        h.observe("192.168.2.10", "lucy", T0 + 5 * DAY + k * 30.0, 700005, True)
    t = T0 + 6 * DAY
    sup = {x: FD.history_support(h.get(x)) for x in ("192.168.2.10", "192.168.2.11", "192.168.2.12")}
    assert sup["192.168.2.10"] == ("lucy", 1.0, 1.0)
    rec = FD.fit_pair(ps, t, support=sup)
    e = rec["table"]["192.168.2.10"]
    assert not e.get("bound") and not e.get("pooled"), e      # node evidence (harmonic burst) < 5
    assert rec["table"]["192.168.2.11"].get("bound")


def test_history_support_ignores_excluded_values_and_disagreeing_nodes():
    h = FD.ValueHistory()
    for d in range(10):
        h.observe("x", "jack", T0 + d * DAY, 700000 + d, True)
    for d in range(4, 10):
        h.observe("x", "rose", T0 + d * DAY + 60, 700000 + d, True)
    assert FD.history_support(h.get("x"), {"rose": "pending"}) == ("jack", 8.0, 8.0)   # HIST_DAYS
    assert FD.history_support(h.get("x")) == ("jack", 8.0, 14.0)
    # a node whose own top value is not the history's top is judged on the node alone
    ps = PN.PairSketch()
    for d in range(3):
        ps.update("x", "rose", T0 + d * DAY, 1.0, 1.0)
    rec = FD.fit_pair(ps, T0 + 3 * DAY, support={"x": ("jack", 10.0, 10.0)})
    assert not rec["table"]["x"].get("pooled") and rec["table"]["x"]["n"] < 5


def test_seeded_history_waits_for_sources_that_log_in_later_in_the_day():
    """The first fit after a node's sketch started tracked only the source that
    had already logged in (kate at 08:50): the seeded history of the two
    others was pruned as 'untracked', and they started again from one day
    (pack O seed 0: .10 / .11 unbound until day 16, kate bound on day 10)."""
    store = MetricStore()
    orc = OracleLearner(store, "finance", T0, ["POST /fin/login", "GET /fin/ledger"], config=CFG)
    eng = BindingEngine()
    hours = {"192.168.2.12": 8.8, "192.168.2.11": 10.8, "192.168.2.10": 12.8}
    for d in range(8):
        for ip, u in sorted(FIN, key=lambda r: hours[r[0]]):
            T = T0 + d * DAY + hours[ip] * 3600
            bb = EV.BatchBuilder("finance")
            bb.add(T - 60, ip, {"http.route": "POST /fin/login", "body.kv.username": u}, 1.0)
            for k in range(12):
                bb.add(T - 50 + k, FIN[k % 3][0], {"http.route": "GET /fin/ledger"}, 1.0)
            b = bb.build(T - 3600, T)
            store.add_batch("finance", EV.EVT_BATCH, T, b)
            orc.learn(b)
            eng.safe_run(_ctx(store, T + 3600))              # P08 fits after each login
    tab = _entries(store, orc)
    assert all((tab.get(ip) or {}).get("bound") for ip, _ in FIN), \
        {ip: (e.get("n"), e.get("pooled")) for ip, e in tab.items()}
