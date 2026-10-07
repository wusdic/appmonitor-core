"""P12 round 6: a family is measured as one system (shared who / behaviour
code, inherited from its most-evidenced member)."""
from __future__ import annotations

from ptree_sim import DAY, daily

from app.engines.behavior import system_profile as SP
from app.engines.behavior.lib import m_ptree as MP
from app.models.schema import ORG, SYSTEM_ENTITY

from test_p12_system_profile import run

IPS = [f"192.168.1.{i}" for i in range(21, 27)]


def staff_day(replica_from: int):
    """Six users, each with its own pages at its own hour (the source predicts
    the behaviour); from day `replica_from` every session goes to 'oa' or to
    its replica 'oa-r2' at random (a load balancer)."""
    def fn(d, t0, r):
        out = []
        if (d % 7) >= 5:
            return out
        for u, ip in enumerate(IPS):
            sysname = "oa" if d < replica_from or r.random() < 0.5 else "oa-r2"
            t = t0 + (8 + u) * 3600 + r.uniform(0, 1200)
            for k in range(8):
                t += r.uniform(20, 240)
                rt = f"GET oa.corp /oa/u{u}/page{k % 4}"
                out.append((t, sysname, ip, {"http.route": rt, "http.method": "GET", "http.host": "oa.corp",
                                             "sess.key": f"s{d}{ip}", "client.stack": "ua",
                                             "net.dst": f"{sysname}:8080"}))
        return out
    return fn


def _gain(sim, s):
    p = sim.st.get_model(MP.tree_key(sim.st, s), SYSTEM_ENTITY, MP.SYSPROF) or {}
    m = p.get("measurements") or {}
    return m.get("who_pred"), m.get("who_pred_n")


def test_a_family_is_measured_like_one_system_with_all_its_traffic():
    """The per-source behaviour gain (who_pred) of a load-balanced pair matches
    a single system that saw every session (round 5: the mean of two half-
    traffic codes, one of them days old, read it lower)."""
    days = 12
    sim, _ = run(daily(staff_day(3), days, seed=5), days)
    fam = sim.st.get_model(ORG, ORG, MP.SYSFAM)
    key = (fam.get("member") or {}).get("oa")
    assert key and key == fam["member"].get("oa-r2")
    one, _ = run(daily(staff_day(10 ** 6), days, seed=5), days)
    g_fam, n_fam = _gain(sim, "oa")
    g_one, n_one = _gain(one, "oa")
    assert g_fam is not None and g_one is not None
    assert abs(n_fam - n_one) <= 0.25 * n_one, (n_fam, n_one)
    assert g_fam[0] >= g_one[0] - 0.25, (g_fam, g_one)


def test_family_members_share_one_who_code_inherited_from_the_seed():
    sim, _ = run(daily(staff_day(3), 10, seed=5), 10)
    key = sim.st.get_model(ORG, ORG, MP.SYSFAM)["member"]["oa"]
    shared = sim.st.get_model(key, SYSTEM_ENTITY, SP.WHO_SHARED)
    assert isinstance(shared, SP.WhoCode)
    assert all(sim.st.get_model(s, SYSTEM_ENTITY, SP.STATE).who is shared for s in ("oa", "oa-r2"))


def test_a_detached_member_gets_its_own_copy_of_the_family_code():
    st_sim, eng = run(daily(staff_day(3), 10, seed=6), 10)
    fam = st_sim.st.get_model(ORG, ORG, MP.SYSFAM)
    key = fam["member"]["oa"]
    shared = st_sim.st.get_model(key, SYSTEM_ENTITY, SP.WHO_SHARED)
    eng._detach(st_sim.st, "oa-r2", key, st_sim.now)
    t2 = st_sim.st.get_model("oa-r2", SYSTEM_ENTITY, SP.STATE)
    t1 = st_sim.st.get_model("oa", SYSTEM_ENTITY, SP.STATE)
    assert t1.who is shared and t2.who is not shared
    assert abs(t2.who.evidence() - shared.evidence()) < 1e-9
