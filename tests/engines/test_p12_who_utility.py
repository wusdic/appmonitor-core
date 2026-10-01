"""P12 who arm utility (progressive.md §6.18.2 as revised in §16): the who arm
decides at which granularity behaviour is conditioned on who, so it is chosen
by the HELD-OUT behaviour gain of each level (WhoCode.observe_beh, bounded
state), with the address code only as a tie-break and as the test that rules
out 'none' for a structured population."""
from __future__ import annotations

from collections import Counter

import numpy as np

from app.engines.behavior.lib import pstrategy as PSt
from app.engines.behavior.lib.phier import Regions
from app.engines.behavior.system_profile import WhoCode

DAY = 86400.0
T0 = 1_741_536_000.0
WCTX = ({}, Counter(), Regions([]), {})


def _feed(wc, day_events, days):
    """day_events(d, r) -> list of (ip, behaviour) for day d; one tick per hour."""
    r = np.random.default_rng(7)
    for d in range(days):
        evs = day_events(d, r)
        for h in range(8):
            part = evs[h::8]
            t = T0 + d * DAY + (9 + h) * 3600.0
            wc.observe(dict(Counter(ip for ip, _ in part)), t, 20000 + d, *WCTX)
            wc.observe_beh(dict(Counter(part)), t, 20000 + d, *WCTX)
    return wc


def _argmax(u, allowed=PSt.WHO_ARMS):
    return max((a for a in allowed if u.get(a) is not None), key=lambda a: u[a])


def test_returning_portal_visitors_are_not_profiled_per_ip():
    """300 public visitors return every few days (stable population, so their
    addresses are coded best per IP) and do what everyone does (behaviour
    independent of the address). The address code alone (the previous
    utility) picks 'ip'; conditioning behaviour on the IP loses held out, so
    the utility picks a coarser level."""
    ips = [f"{1 + i % 200}.{i % 7}.{i % 13}.{10 + i}" for i in range(300)]
    acts = [("GET portal /", True, 10), ("GET portal /news", True, 11), ("POST portal /login", True, 12),
            ("GET portal /search", True, 13)]

    def ev(d, r):
        out = []
        for ip in r.choice(ips, 120, replace=False):
            for _ in range(int(r.integers(2, 6))):
                out.append((str(ip), acts[int(r.integers(0, len(acts)))]))
        return out
    wc = _feed(WhoCode(), ev, 14)
    bits, _ = wc.bits()
    gain, n = wc.beh_gain()
    assert n > 1000
    old = PSt.who_utilities(bits)
    new = PSt.who_utilities(bits, gain)
    assert _argmax(old) == "ip", (bits, old)
    assert gain[0] < -0.1, gain                     # per-IP behaviour models lose held out
    assert _argmax(new) != "ip", (gain, new)


def test_level_that_predicts_behaviour_is_chosen_over_finer_ones():
    """Three departments, each in its own /24, each with its own actions; the
    members of a department behave alike, and a third of each department's
    addresses are re-leased every day. Conditioning on the /24 (here the
    department) predicts behaviour from the first event of a new address; the
    per-IP level must relearn each new address."""
    depts = {k: [f"10.{k}.0.{i}" for i in range(1, 200)] for k in (1, 2, 3)}
    acts = {k: [(f"POST app /d{k}/a{j}", True, 9 + k) for j in range(3)] for k in (1, 2, 3)}

    def ev(d, r):
        out = []
        for k, pool in depts.items():
            for ip in r.choice(pool, 12, replace=False):
                for _ in range(4):
                    out.append((str(ip), acts[k][int(r.integers(0, 3))]))
        return out
    wc = _feed(WhoCode(), ev, 10)
    bits, _ = wc.bits()
    gain, _ = wc.beh_gain()
    assert gain[1] > 1.0, gain                      # /24 = department: ~log2(3) bits/event
    assert gain[1] > gain[0] + 0.2, gain
    u = PSt.who_utilities(bits, gain)
    assert _argmax(u) == "prefix", u


def test_closed_population_with_identical_behaviour_is_not_none():
    """Five fixed addresses use a system identically: no level predicts
    behaviour (all gains <= 0), but the population is closed (its address code
    saves ~27 bits/event), so 'none' (the address is not a feature) is ruled
    out; a random population (every event a new address) leaves 'none'."""
    ips = [f"192.168.1.{i}" for i in (21, 23, 30, 31)] + ["10.168.7.121"]
    acts = [(f"GET oa /p{j}", True, 9) for j in range(5)]

    def same(d, r):
        return [(ip, acts[int(r.integers(0, 5))]) for ip in ips for _ in range(6)]

    def rand(d, r):
        return [(f"{int(r.integers(1, 223))}.{int(r.integers(0, 256))}.{int(r.integers(0, 256))}."
                 f"{int(r.integers(1, 255))}", acts[int(r.integers(0, 5))]) for _ in range(40)]
    wc = _feed(WhoCode(), same, 7)
    u = PSt.who_utilities(*[wc.bits()[0], wc.beh_gain()[0]])
    assert u["none"] < -1.0 and _argmax(u) != "none", u
    wr = _feed(WhoCode(), rand, 7)
    ur = PSt.who_utilities(wr.bits()[0], wr.beh_gain()[0])
    assert ur["none"] == 0.0 and _argmax(ur) == "none", (wr.bits()[0], ur)


def test_behaviour_code_state_is_bounded():
    """The (item, behaviour) statistics are Space-Saving capped: 20 000 distinct
    addresses x 50 behaviours leave the tracker's size bounded."""
    acts = [(f"GET x /{j}", True, 9) for j in range(50)]

    def ev(d, r):
        return [(f"10.{d}.{i // 256}.{i % 256}", acts[i % 50]) for i in range(2000)]
    wc = _feed(WhoCode(), ev, 10)
    assert len(wc.bm) <= WhoCode.KB_MARG
    assert all(len(x) <= WhoCode.KB_PAIR for x in wc.bp)
    assert all(len(x) <= WhoCode.KB_ITEM for x in wc.bi)


def test_bindings_not_yet_judged_are_unmeasured_not_zero():
    """P08's arm is measured from its per-node records; a pair record that has
    not judged any source yet (no source with n_bind events) is not a gain of
    0 (pack O seed 0: finance's bindings were switched off on day 7, before its
    users had 5 logins, and stayed off for the rest of the run)."""
    from types import SimpleNamespace as NS
    from app.engines.behavior.system_profile import fitted_gain
    node = NS(parent=None, mass_at=lambda t: 100.0)
    leaf = NS(parent=0, mass_at=lambda t: 10.0)
    ptm = NS(kinds={0: NS(root=0, nodes={0: node, 1: leaf})})
    young = {"nodes": {0: {1: {"pairs": {"p": {"gain": 0.0, "fd": {"judged": 0}}}}}}}
    assert fitted_gain(young, ptm, 0.0) is None
    judged0 = {"nodes": {0: {1: {"pairs": {"p": {"gain": 0.0, "fd": {"judged": 3}}}}}}}
    assert fitted_gain(judged0, ptm, 0.0) == 0.0          # judged and nothing binds: a real 0
    bound = {"nodes": {0: {1: {"pairs": {"p": {"gain": 1.5, "fd": {"judged": 3}}}}}}}
    assert fitted_gain(bound, ptm, 0.0) == 0.15
    assert fitted_gain({"nodes": {}}, ptm, 0.0) is None


def test_switch_margin_ignores_the_trend_both_arms_share():
    """While a system's models are learnt both arms' utilities rise together;
    the margin is the noise of their DIFFERENCE, not of their levels (pack O
    seed 0 finance: the old margin was 1.5 bits/event against a steady 0.17
    advantage of the per-IP arm, adopted 9 days late)."""
    cur = [-1.25, -0.80, -0.43, -0.10, 0.20, 0.45, 0.62]
    new = [c + 0.17 + e for c, e in zip(cur, [0.01, -0.01, 0.0, 0.01, -0.01, 0.0, 0.01])]
    m = PSt.switch_margin(cur, new)
    assert m < 0.17, m
    old = 2.0 * float(np.sqrt(np.var(cur, ddof=1) + np.var(new, ddof=1)))
    assert old > 1.0                                   # what the level variance gave
    sw = PSt.Switcher("prefix")
    for d in range(7):
        sw._record({"prefix": cur[d], "ip": new[d]})
    out = [sw.step("ip", {"prefix": cur[-1], "ip": new[-1]}, ["ip", "prefix"], 100 + d)[0]
           for d in range(3)]
    assert out[-1] == "ip"


def test_unjudged_fitter_is_unmeasured_only_while_the_system_is_young(monkeypatch):
    from types import SimpleNamespace as NS
    from app.engines.behavior import system_profile as SP
    from app.engines.behavior.lib import m_ptree as MP
    from app.core.store import MetricStore
    from helpers import ctx
    st = MetricStore()
    now = T0 + 20 * DAY
    tr = SP.SysTracker("portal", T0)
    st.put_model("portal", "__system__", SP.STATE, tr, ts=now)
    st.put_model("portal", "__system__", MP.PBIND, {"updated": now, "nodes": {},
                                                    "gain": {"bits_per_event": 0.0, "ms": 5.0}}, ts=now)
    monkeypatch.setattr(SP.MP, "get_ptree", lambda store, key: NS(kinds={}))
    eng = SP.SystemProfileEngine()
    c = ctx(st, now, window_s=3600.0, config={"progressive": {"enabled": True}})
    tr.days_seen = 3
    assert "P08" not in eng.measurements(c, "portal", ["portal"], {"volume": {}}, now)
    tr.days_seen = 10
    assert eng.measurements(c, "portal", ["portal"], {"volume": {}}, now)["P08"]["gain"] == 0.0


def test_hedge_rounds_are_the_last_completed_day_not_the_seven_day_sum():
    """Hedge needs one loss per round; feeding the 7-day sums every day counted
    each day seven times and the leader followed a change a week late (pack O
    seed 2 finance: the per-IP arm led the day's gains from day 10, the
    leader turned on day 13). With a day's figures present (and enough
    evidence) the who utilities are the day's."""
    bits7, pred7 = [3.0, 9.5, 16.0, 32.0, 32.0], [0.0, 0.3, -0.5, -0.5, -0.5]     # prefix best over 7 d
    bitsd, predd = [3.0, 9.5, 16.0, 32.0, 32.0], [0.9, 0.7, -0.5, -0.5, -0.5]     # ip best today
    meas = {"who": bits7, "who_n": 500.0, "who_pred": pred7, "who_pred_n": 500.0,
            "who_day": bitsd, "who_day_n": 80.0, "who_pred_day": predd, "who_pred_day_n": 80.0}
    st = PSt.new_state()
    out = PSt.decide(st, {"ip_info": 0.5, "churn": 0.05}, meas, 20010, "finance")
    U = {a: v["U"] for a, v in out["arms"]["who"].items()}
    assert U == PSt.who_utilities(bitsd, predd)
    assert U["ip"] > U["prefix"]
    # a day with too little behaviour evidence: the 7-day figures
    meas2 = dict(meas, who_pred_day_n=5.0, who_day_n=5.0)
    out2 = PSt.decide(PSt.new_state(), {"ip_info": 0.5, "churn": 0.05}, meas2, 20010, "finance")
    U2 = {a: v["U"] for a, v in out2["arms"]["who"].items()}
    assert U2 == PSt.who_utilities(bits7, pred7)
    # few addresses (pack O finance: 9-21 address units a day, 114-173
    # behaviour units): the day's behaviour gains with the 7-day address code
    bits7b = [2.5, 9.0, 16.0, 32.0, 32.0]
    meas3 = dict(meas, who=bits7b, who_day_n=12.0)
    out3 = PSt.decide(PSt.new_state(), {"ip_info": 0.5, "churn": 0.05}, meas3, 20010, "finance")
    U3 = {a: v["U"] for a, v in out3["arms"]["who"].items()}
    assert U3 == PSt.who_utilities(bits7b, predd)
