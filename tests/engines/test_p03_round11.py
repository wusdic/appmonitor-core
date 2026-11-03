"""P03 round 11 (bounds owner, item 3; progressive.md §16.18.12 item 9): a
source P11 has made a member of the node's stated who - the member it saw
move to this address (who_groups 'readdressed': the old address silent
since; P14's who clause names the member at the new address) - is not judged
foreign at the node, and its rows are not damped.

O-real seed 0 (R3): the finance clerk 192.168.2.11 moved to .51 on day 9.
At the clerk's individual finance nodes the department group had no OTHER
member with standing (group_outsider), so .51 was 'outsider_group', judged
foreign and learned damped (0.1) - and P04's suspect test then kept it out of
the who summaries the statements are rendered from. The colleague rule's
guard (A1: a group member WITHOUT a place of its own at someone else's
individual node stays an outsider) is unchanged."""
from __future__ import annotations

from app.engines.behavior import conformity as CF
from app.engines.behavior.lib import m_ptree as MP
from app.models.schema import ORG

from test_p03_conformity import FIN, Fx, workdays

LEDGER = "GET fin /fin/ledger/close"
OLD, NEW, STRANGER = "192.168.2.11", "192.168.2.51", "192.168.2.77"


def _clerk_node(readdressed: bool):
    """The ledger-close node: the clerk's own (only .11 uses it, 21 workdays);
    P11's 财务部 group G2 = .10, .11, .12 (+ .51, + .77)."""
    fx = Fx()
    for d in workdays(21):
        for k in range(2):
            fx.learn("finance", LEDGER, OLD, d + (16 * 60 + 7 * k) * 60.0)
    wg = dict(fx.st.get_model(ORG, ORG, MP.WHO_GROUPS))
    mem = sorted(set(FIN) | {NEW, STRANGER})
    wg["groups"] = dict(wg["groups"], G2=dict(wg["groups"].get("G2") or {"id": "G2", "name": "G2"}, members=mem))
    wg["ip2g"] = dict(wg["ip2g"], **{NEW: "G2", STRANGER: "G2"})
    if readdressed:
        wg["readdressed"] = {OLD: {"new": NEW, "t": workdays(21)[-1], "g": "G2"}}
    fx.st.put_model(ORG, ORG, MP.WHO_GROUPS, wg)
    return fx


def _read(fx, ip, t):
    _, asg = fx.score("finance", [(t, ip, {"http.route": LEDGER, "http.method": "GET"})])
    return float(asg.dense("p_who", 1.0)[0]), float(asg.dense("damp", 1.0)[0])


def test_readdressed_member_of_the_stated_who_is_not_damped():
    fx = _clerk_node(readdressed=True)
    t = workdays(23)[-1] + 16 * 3600.0
    nd = fx.node("finance", LEDGER)
    lvl, heavy, _U = CF._who_closed(nd, t)
    assert lvl == 0 and heavy == {OLD}
    p_who, damp = _read(fx, NEW, t)
    # round 10: p_who = U (foreign, 'outsider_group'), damp 0.1
    assert p_who == 1.0 and damp == 1.0
    assert not [e for e in fx.violations(NEW) if e.extra["type"] == "who"]
    # the node's reference statement (yesterday's who list) still names the
    # member at its old address: not a second, contradicting anchor
    nd.ref = dict(nd.ref or {}, who=[OLD])
    p_who, damp = _read(fx, NEW, t + 600.0)
    assert p_who == 1.0 and damp == 1.0


def test_a_group_member_without_a_place_at_the_node_stays_an_outsider():
    # no readdress: the same group membership alone does not make a member of
    # a colleague's individual node (A1's guard)
    fx = _clerk_node(readdressed=False)
    t = workdays(23)[-1] + 16 * 3600.0
    p_who, damp = _read(fx, NEW, t)
    assert p_who < 1.0 and damp < 1.0
    # and with the readdress, ANOTHER group member is still an outsider
    fx = _clerk_node(readdressed=True)
    p_who, damp = _read(fx, STRANGER, t)
    assert p_who < 1.0 and damp < 1.0
    w = [e for e in fx.violations(STRANGER) if e.extra["type"] == "who"]
    assert w and "outsider_group" in w[0].extra["flags"]


def test_readdressed_source_is_one_entry_of_a_set_binding():
    """The department's client stack (a SET reverse binding tls.ja3 -> {.11,
    .12}): the moved clerk using it while .12 is active is not concurrent use."""
    fx = _clerk_node(readdressed=True)
    nid = fx.node("finance", LEDGER).id
    rev = {"x": "tls.ja3", "y": "net.src", "dir": "rev", "fd": {"g3": 0.0, "holds": True},
           "table": {"J1": {"set": [OLD, "192.168.2.12"], "shares": [0.45, 0.55], "U": 0.006, "n": 80.0,
                            "top": "192.168.2.12"}}}
    fx.put("finance", MP.PBIND, nid, pairs={"tls.ja3->net.src": rev})
    t = workdays(23)[-1] + 16 * 3600.0
    ev = {"http.route": LEDGER, "http.method": "GET", "tls.ja3": "J1"}
    fx.score("finance", [(t - 300.0, "192.168.2.12", ev)])
    _, asg = fx.score("finance", [(t, NEW, ev)])
    assert float(asg.dense("damp", 1.0)[0]) == 1.0
    flags = {f for e in fx.violations(NEW) for f in (e.extra.get("flags") or [])}
    assert "concurrent_use" not in flags and "foreign_source" not in flags
    # a stranger with the same client stack is still concurrent use
    fx.score("finance", [(t + 60.0, STRANGER, ev)])
    flags = {f for e in fx.violations(STRANGER) for f in (e.extra.get("flags") or [])}
    assert "concurrent_use" in flags


def test_readdress_notes_last_the_local_day_and_are_bounded():
    """(round 11) ConfState.readdr_note / readdr_of: the source a new address
    was a readdress candidate of is remembered for the rest of that local day
    (lib/pfd.check_reverse's `readdr_of`), not after it; a state saved before
    the notes existed reads none; at most READDR_NOTES are kept."""
    st = CF.ConfState(16, 16)
    assert st.readdr_of("finance", NEW, 100) is None             # no notes attribute yet
    st.readdr_note("finance", NEW, OLD, 100)
    assert st.readdr_of("finance", NEW, 100) == OLD
    assert st.readdr_of("oa", NEW, 100) is None
    assert st.readdr_of("finance", NEW, 101) is None
    for i in range(CF.READDR_NOTES + 10):
        st.readdr_note("finance", f"10.9.{i // 250}.{i % 250}", OLD, 100)
    assert len(st.readdr_notes) == CF.READDR_NOTES
    st.readdr_note("finance", STRANGER, OLD, 101)                 # a new day drops the old notes
    assert list(st.readdr_notes) == [("finance", STRANGER)]
    assert st.readdr_of("finance", NEW, 101) is None
