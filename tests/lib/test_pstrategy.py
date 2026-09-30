"""lib/pstrategy: scenario-adaptive strategy selection (progressive.md §6.18)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.engines.behavior.lib import pstrategy as PSt


def test_utility_single_currency():
    assert PSt.utility(0.5, 100.0) == pytest.approx(0.5 - 0.1)
    assert PSt.utility(None, 10.0) is None
    assert PSt.utility(0.2, None) == pytest.approx(0.2)


def test_who_arms_and_utilities():
    b = PSt.who_arm_bits([5.0, 12.0, 17.0, 3.0, 9.0])
    assert b["ip"] == 5.0 and b["prefix"] == 12.0 and b["grp"] == 3.0 and b["none"] == 32.0
    U = PSt.who_utilities([5.0, 12.0, 17.0, 3.0, 9.0])
    assert U["none"] == 0.0 and U["grp"] == pytest.approx(29.0)
    # a missing level is never attractive
    U2 = PSt.who_utilities([5.0, float("nan"), 17.0, None, 9.0])
    assert U2["grp"] == -PSt.U_CLIP and U2["prefix"] == pytest.approx(15.0)


def test_who_preconditions():
    ok = PSt.who_preconditions({"ip_info": {0: 0.4}, "churn": 0.05, "grp_cover": 0.8, "reg_cover": 0.0})
    assert ok["ip"][0] and ok["grp"][0] and not ok["reg"][0] and ok["prefix"][0] and ok["none"][0]
    churn = PSt.who_preconditions({"ip_info": {0: 0.4}, "churn": 0.6})
    assert not churn["ip"][0] and "churn" in churn["ip"][1]
    noinfo = PSt.who_preconditions({"ip_info": {0: 0.01}, "churn": 0.0})
    assert not noinfo["ip"][0]
    snat = PSt.who_preconditions({"snat": True, "ip_info": {0: 0.9}, "grp_cover": 1.0})
    assert [a for a, (ok_, _) in snat.items() if ok_] == ["none"]


def test_content_preconditions():
    ch = {"payload_vis": {"body": 0.3}, "sess_ident": 0.9, "route_card": 12, "sess_key_cov": 0.0}
    c = PSt.content_preconditions(ch, "ip")
    assert c["P07"][0] and c["P08"][0] and c["P10"][0]
    c = PSt.content_preconditions(ch, "none")
    assert not c["P08"][0]                         # who none and no session key
    opaque = PSt.content_preconditions({"payload_vis": {"body": 0.0}, "sess_ident": 0.1,
                                        "route_card": 2}, "grp")
    assert not opaque["P07"][0] and not opaque["P08"][0] and not opaque["P10"][0]


def test_hedge_leader_and_roundtrip():
    h = PSt.Hedge(PSt.WHO_ARMS)
    for _ in range(5):
        h.update({"ip": 20.0, "grp": 25.0, "prefix": 10.0, "reg": 5.0, "none": 0.0})
    assert h.leader() == "grp"
    assert h.leader(["ip", "prefix"]) == "ip"
    p = h.probs()
    assert abs(sum(p.values()) - 1.0) < 1e-9 and p["grp"] > 0.99
    h2 = PSt.Hedge.from_dict(h.to_dict())
    assert h2.leader() == "grp" and h2.n == 5
    # clipping: an unbounded ratio cannot dominate in one day
    h3 = PSt.Hedge(("on", "off"))
    h3.update({"on": 1e9, "off": 0.0})
    assert h3.logw["off"] == pytest.approx(-PSt.ETA * PSt.U_CLIP)
    # relative utilities: arms saving 20 and 25 bits are still told apart
    h4 = PSt.Hedge(("ip", "grp"))
    h4.update({"ip": 20.0, "grp": 25.0})
    assert h4.logw["ip"] == pytest.approx(-PSt.ETA * 5.0)


def test_switcher_hysteresis_no_flapping_under_noise():
    """Two arms with equal mean utility and noisy daily measurements: the
    incumbent is kept (no flapping) over 200 days and 5 seeds."""
    for seed in range(5):
        r = np.random.default_rng(seed)
        sw = PSt.Switcher()
        h = PSt.Hedge(("a", "b"))
        switches = 0
        for d in range(200):
            U = {"a": 1.0 + r.normal(0, 0.1), "b": 1.0 + r.normal(0, 0.1)}
            h.update(U, eta=0.5)
            _, ch, _ = sw.step(h.leader(), U, ["a", "b"], d)
            switches += int(ch and d > 0)
        assert switches <= 2, (seed, switches)


def test_switcher_switches_on_a_real_difference():
    sw = PSt.Switcher("a")
    days = []
    for d in range(10):
        U = {"a": 0.0, "b": 0.3}
        cur, ch, _ = sw.step("b", U, ["a", "b"], d)
        if ch:
            days.append(d)
    assert days == [PSt.SWITCH_DAYS - 1]
    assert sw.cur == "b"


def test_switcher_precondition_is_hard():
    sw = PSt.Switcher("ip")
    cur, ch, why = sw.step("prefix", {"ip": 30.0, "prefix": 10.0}, ["prefix", "none"], 3)
    assert cur == "prefix" and ch and why == "precondition"


def test_adaptive_margin_deterministic_cost_only_difference():
    """A fitter whose gain is exactly zero differs from 'off' only by its cost:
    the margin shrinks to >= 0.002 bits/event and the arm is switched off even
    at 5 µs/event (0.005 bits/event), below the fixed 0.05."""
    assert PSt.switch_margin([-0.005] * 5, [0.0] * 5) == PSt.SWITCH_MARGIN_MIN
    assert PSt.switch_margin([0.0, 0.3, -0.2, 0.1, 0.2], [0.1] * 5) > PSt.SWITCH_MARGIN
    assert PSt.switch_margin([0.1, 0.2], [0.0] * 5) == PSt.SWITCH_MARGIN
    st = PSt.new_state()
    ch = {"payload_vis": {"body": 0.5}, "sess_ident": 0.9, "route_card": 20, "ip_info": {0: 0.5},
          "churn": 0.0, "grp_cover": 0.0}
    chosen = []
    for d in range(14):
        meas = {"who": [4.0, 12.0, 17.0, 32.0, 32.0], "who_n": 100.0,
                "P08": {"gain": 0.0, "cost": 5.0}, "P07": {"gain": 0.8, "cost": 20.0}}
        out = PSt.decide(st, ch, meas, 1000 + d, "portal", explore=False)
        chosen.append(out["chosen"]["P08"])
    assert chosen[0] == "on" and chosen[-1] == "off", chosen
    assert out["chosen"]["P07"] == "on"


def test_decide_departmental_oa_vs_random_portal_vs_opaque_vs_snat():
    oa = {"payload_vis": {"body": 0.4, "http": 1.0}, "sess_ident": 0.95, "route_card": 30,
          "ip_info": {0: 0.6, 3: 0.7}, "churn": 0.01, "grp_cover": 0.95, "reg_cover": 0.0}
    portal = {"payload_vis": {"body": 0.3, "http": 1.0}, "sess_ident": 0.9, "route_card": 15,
              "ip_info": {0: 0.01}, "churn": 0.5, "grp_cover": 0.02, "reg_cover": 1.0}
    mail = {"payload_vis": {"body": 0.0, "http": 0.0, "tls_opaque": 1.0}, "sess_ident": 0.2,
            "route_card": 0, "ip_info": {0: 0.3}, "churn": 0.01, "grp_cover": 0.9, "reg_cover": 0.0}
    snat = dict(oa, snat=True)
    res = {}
    for name, ch, who in (("oa", oa, [6.0, 13.0, 18.0, 4.0, 33.0]),
                          ("portal", portal, [34.0, 33.0, 20.0, 40.0, 11.0]),
                          ("mail", mail, [7.0, 14.0, 19.0, 5.0, 33.0]),
                          ("snat", snat, [1.0, 9.0, 16.0, 1.0, 33.0])):
        st = PSt.new_state()
        for d in range(10):
            meas = {"who": who, "who_n": 500.0,
                    "P07": {"gain": 0.6, "cost": 30.0} if ch["payload_vis"]["body"] > 0 else None,
                    "P08": ({"gain": 1.5, "cost": 30.0} if name == "oa" else {"gain": 0.0, "cost": 20.0})
                    if ch["payload_vis"]["body"] > 0 else None,
                    "P10": {"gain": 0.4, "cost": 40.0} if ch["sess_ident"] >= 0.6 else None,
                    "n_nodes": 200, "ev_day": 5000}
            out = PSt.decide(st, ch, meas, 2000 + d, name, explore=False)
        res[name] = out["chosen"]
    assert res["oa"]["who"] in ("ip", "grp") and res["oa"]["P07"] == "on" and res["oa"]["P08"] == "on" \
        and res["oa"]["P10"] == "on"
    assert res["portal"]["who"] in ("prefix", "reg", "none") and res["portal"]["P08"] == "off"
    assert res["mail"]["P07"] == "off" and res["mail"]["P08"] == "off" and res["mail"]["content"] == "off"
    assert res["snat"]["who"] == "none"
    for c in res.values():                      # aliases the consumers read
        assert c["p07"] == c["P07"] and c["p10"] == c["P10"]


def test_probe_day_turns_an_off_arm_on_for_one_day():
    st = PSt.new_state()
    ch = {"payload_vis": {"body": 0.5}, "sess_ident": 0.9, "route_card": 20, "ip_info": {0: 0.5},
          "churn": 0.0}
    days_on = []
    for d in range(60):
        meas = {"who": [4.0, 12.0, 17.0, 32.0, 32.0], "who_n": 100.0,
                "P08": {"gain": 0.0, "cost": 30.0}}
        out = PSt.decide(st, ch, meas, 3000 + d, "sysx", explore=True)
        if d > 20 and out["chosen"]["P08"] == "on":
            days_on.append(d)
    # off after the switch; on only on the probe days (1 in 14)
    assert 2 <= len(days_on) <= 3
    assert all(PSt.probe_day("sysx", 3000 + d) for d in days_on)


def test_budgeted_ucb():
    u = PSt.BudgetedUCB()
    assert u.index("on") == math.inf
    for _ in range(20):
        u.observe("on", 0.001, 200.0)              # 0.001 bits for 200 µs: not worth it
    assert not u.worth_running("on")
    u2 = PSt.BudgetedUCB()
    for _ in range(20):
        u2.observe("on", 0.5, 50.0)
    assert u2.worth_running("on")
    assert PSt.BudgetedUCB.from_dict(u2.to_dict()).mean("on")[0] == pytest.approx(u2.mean("on")[0])


def test_knapsack_packing():
    items = [("P07", 0.6, 30.0), ("P08", 1.5, 30.0), ("P10", 0.4, 200.0), ("bad", 0.01, 100.0)]
    assert set(PSt.pack_knapsack(items, None)) == {"P07", "P08", "P10"}
    assert PSt.pack_knapsack(items, 60.0) == ["P08", "P07"]
    assert PSt.pack_knapsack(items, 10.0, mandatory=["P10"]) == ["P10"]


def test_bimodality_and_heaps():
    x = np.linspace(0, 6, 48)
    uni = np.exp(-0.5 * ((x - 3.0) / 0.5) ** 2) * 1000
    bi = uni * 0 + np.exp(-0.5 * ((x - 1.0) / 0.3) ** 2) * 1000 + np.exp(-0.5 * ((x - 4.5) / 0.3) ** 2) * 600
    assert PSt.bimodality_coefficient(uni, x) < PSt.BC_BIMODAL
    assert PSt.bimodality_coefficient(bi, x) > PSt.BC_BIMODAL
    assert PSt.bimodality_coefficient([1, 2], [0, 1]) is None
    pts = [(10.0 ** k, 3.0 * (10.0 ** k) ** 0.5) for k in range(1, 6)]
    assert PSt.heaps_beta(pts) == pytest.approx(0.5, abs=1e-6)


def test_tier_and_earned_cap_and_applicability():
    assert PSt.recommend_tier(10, 50) == "XS"
    assert PSt.recommend_tier(40, 50) == "S"                     # grown small system is not pruned back
    assert PSt.recommend_tier(100, 5000) == "S"
    assert PSt.recommend_tier(600, 5000) == "M"
    assert PSt.recommend_tier(3000, 5000) == "L"
    assert PSt.recommend_tier(160, 5000, cur="M") == "M"        # hysteresis: not yet below half of S
    assert PSt.recommend_tier(100, 5000, cur="M") == "S"
    assert [PSt.earned_cap(n) for n in (0, 8, 9, 100, 1000)] == [8, 8, 32, 128, 128]
    ap = PSt.b_applicability({"sess_ident": 0.1, "automation": 0.0, "stack_vis": 0.9,
                              "upload_routes": True})
    assert ap["B10"] == "class" and ap["B11"] == "class" and ap["B09"] == "on" and ap["B13"] == "on"
    ap2 = PSt.b_applicability({"sess_ident": 0.9, "automation": 0.3}, vetoed=["sequence"])
    assert ap2["B10"] == "class" and ap2["B11"] == "on"
    assert PSt.label_veto({"sequence": (0, 6), "timing": (3, 5), "budget": (0, 4)}) == ["sequence"]


def test_arm_turns_on_when_its_precondition_starts_to_hold():
    """Day 1 of a system: no payload seen yet -> P07 off by precondition; once
    bodies are visible P07 is switched on the same day (not after a probe)."""
    st = PSt.new_state()
    base = {"sess_ident": 0.9, "route_card": 20, "ip_info": {0: 0.5}, "churn": 0.0}
    out = PSt.decide(st, dict(base, payload_vis={"body": 0.0}), {}, 10, "oa", explore=False)
    assert out["chosen"]["P07"] == "off" and out["chosen"]["P08"] == "off"
    out = PSt.decide(st, dict(base, payload_vis={"body": 0.4}), {}, 11, "oa", explore=False)
    assert out["chosen"]["P07"] == "on" and out["chosen"]["P08"] == "on"
    assert {c[0] for c in out["changed"]} >= {"P07", "P08"}


def test_budget_packing_when_the_cpu_budget_binds():
    st = PSt.new_state()
    ch = {"payload_vis": {"body": 0.5}, "sess_ident": 0.9, "route_card": 20, "ip_info": {0: 0.5},
          "churn": 0.0}
    meas = {"who": [4.0, 12.0, 17.0, 32.0, 32.0], "who_n": 100.0,
            "P07": {"gain": 2.0, "cost": 50.0}, "P08": {"gain": 0.2, "cost": 400.0},
            "P10": {"gain": 1.0, "cost": 50.0}}
    out = PSt.decide(st, ch, dict(meas, cpu_usage_share=0.3), 100, "x", explore=False)
    assert (out["chosen"]["P07"], out["chosen"]["P08"], out["chosen"]["P10"]) == ("on", "on", "on")
    out = PSt.decide(st, ch, dict(meas, cpu_usage_share=4.0), 101, "x", explore=False)
    # 500 µs of optional work, 125 allowed: the two best gain/cost arms stay
    assert (out["chosen"]["P07"], out["chosen"]["P08"], out["chosen"]["P10"]) == ("on", "off", "on")
    assert "budget" in out["reasons"]["P08"]
