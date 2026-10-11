"""P11 who groups / P14 rendering, round 6 (time / views owner): a member P03
saw move to a new address (readdress_candidate) is remembered as a move
(who_groups 'readdressed') while the old address stays silent, and the views
state the member at its new address. Fails on the round-5 code."""
from __future__ import annotations

from types import SimpleNamespace as NS

from test_p11_who_groups import OrgSim, dept_ips

from app.engines.behavior.lib import prender as PR
from app.engines.behavior.lib import pnode as PN


def _moved_sim():
    sim = OrgSim()
    sim.days(4)
    ips = dict(dept_ips())
    ips["FIN"] = ["192.168.2.10", "192.168.2.51", "192.168.2.12"]
    sim.readdr = [("192.168.2.51", "192.168.2.11")]
    sim.plan_day(sim.now + 8 * 3600.0, ips=ips)
    for _ in range(24):
        sim.tick()
    return sim, ips


def test_a_readdress_is_remembered_while_the_old_address_is_silent():
    """O-real seed 0, R3: 财务部's 192.168.2.11 moved to .51 on day 9; P11 put
    .51 in the group but nothing said .11 had left it."""
    sim, ips = _moved_sim()
    rd = sim.model().get("readdressed") or {}
    assert rd.get("192.168.2.11", {}).get("new") == "192.168.2.51"
    sim.days(3, ips=ips)
    assert "192.168.2.11" in (sim.model().get("readdressed") or {})


def test_a_readdress_is_forgotten_when_the_old_address_returns():
    sim, ips = _moved_sim()
    back = dict(ips)
    back["FIN"] = ["192.168.2.10", "192.168.2.11", "192.168.2.51", "192.168.2.12"]
    sim.days(3, ips=back)                      # spans a workday after the weekend
    assert "192.168.2.11" not in (sim.model().get("readdressed") or {})


def test_an_address_list_states_a_moved_member_at_its_new_address():
    """The finance login node's who summary still holds .11's mass (and not
    yet .51's) after the move: the statement lists .10, .12, .51."""
    who = PN.WhoSummary()
    t = 1_790_035_200.0
    for d in range(8):
        for ip in ("192.168.2.10", "192.168.2.11", "192.168.2.12"):
            for k in range(3):
                ts = t + d * 86400.0 + k * 60.0
                who.update([ip, ip.rsplit(".", 1)[0] + ".0/24", "192.168.0.0/16", None, None], ip, ts, 1.0, 1.0)
    now = t + 8 * 86400.0
    ev0, *_ = PR.who_block(who, now, 8, {}, {})
    ev1, zh, _en, _c = PR.who_block(who, now, 8, {}, {}, readdr={"192.168.2.11": "192.168.2.51"})
    assert ev0["level"] == "ip" and "192.168.2.11" in ev0["items"]
    assert ev1["items"] == ["192.168.2.10", "192.168.2.12", "192.168.2.51"]
    assert "192.168.2.51" in zh and "192.168.2.11" not in zh
