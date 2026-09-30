"""P02 AttributeRegistry (behavior.attr_registry): open attribute space —
registration on first sight, typing, statistics from learned rows of t - D
only, A_ev per event, schema change (docs/lib3/progressive.md §5.3, §6.3, card P02)."""
from __future__ import annotations

import numpy as np

from ptree_sim import DAY, MON, Sim, daily

from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV


def _day(d, t0, rng, with_hdr=True):
    ev = []
    for i in range(60):
        ts = t0 + rng.uniform(0, DAY)
        a = {"http.route": f"GET x /r{i % 4}", "net.bytes_up": float(rng.lognormal(7, 1.0)),
             "http.status": int(rng.choice([200, 302, 404])),
             "body.keys": frozenset({"username", "password"}),
             "hdr.cookie2": f"a={i}&b={i % 3}",
             "meta.peer": f"10.0.{i % 4}.{i % 7}"}
        if with_hdr:
            a["hdr.x-client-ver"] = "5.2.1"
        ev.append((ts, "oa", f"192.168.1.{i % 20}", a))
    return ev


def test_registration_typing_and_new_events():
    sim = Sim(sel={"targets_sys": {0: []}, "split_cands": {0: []}})
    sim.add(daily(_day, 2, seed=1))
    sim.run_until(MON + 2 * DAY)
    reg = MP.get_registry(sim.st, "oa")
    assert reg.get("net.bytes_up").type == "numeric" and reg.get("net.bytes_up").hier.get("log")
    assert reg.get("meta.peer").type == "ip"
    assert reg.get("body.keys").type == "set"
    assert reg.get("http.status").type in ("categorical", "ordinal")
    assert reg.get("hdr.cookie2").parse_as == "form"
    assert reg.get("net.bytes_up").hier.get("edges") is not None       # first refresh after 1 h
    new = [e for e in sim.events("attribute_new")]
    assert {e.extra["attribute"] for e in new} >= {"net.bytes_up", "meta.peer", "hdr.x-client-ver"}


def test_statistics_use_rows_of_t_minus_D_only():
    sim = Sim(sel={"targets_sys": {0: []}, "split_cands": {0: []}})
    t = MON + 3600.0 * 10 + 1.0
    sim.add([(t, "oa", "10.0.0.1", {"http.route": "GET x /a", "net.bytes_up": 10.0})])
    sim.run_until(MON + 3600.0 * 10 + 900.0)
    reg = MP.get_registry(sim.st, "oa")
    rec = reg.get("net.bytes_up")
    assert rec is not None                                              # registered at once
    assert float(rec.pres.read(sim.now)[0]) == 0.0                      # but not learned yet
    D = EV.learn_delay_s(sim.dt, sim.cfg)
    sim.run_until(sim.now + D + sim.dt)
    assert float(rec.pres.read(sim.now)[0]) > 0.0


def test_at_most_a_ev_attribute_updates_per_event():
    sim = Sim(sel={"targets_sys": {0: []}, "split_cands": {0: []}, "roles": {}})
    t0 = MON + 9 * 3600.0
    ev = []
    for i in range(50):
        a = {f"meta.f{k:03d}": float(k + i % 3) for k in range(200)}
        ev.append((t0 + 10 * i, "oa", "10.0.0.1", a))
    sim.add(ev)
    sim.run_until(t0 + 6 * 3600.0)
    reg = MP.get_registry(sim.st, "oa")
    assert len(reg) >= 200
    total_rows = sum(float(r.te.read(sim.now)[0]) for n, r in reg.records.items() if n.startswith("meta.f"))
    # 50 learned events x A_ev = 32 updates (decay makes it slightly smaller)
    assert 0.5 * 50 * 32 <= total_rows <= 50 * 32 * 1.3


def test_schema_change_declared_gone_within_one_normal_day():
    sim = Sim(sel={"targets_sys": {0: []}, "split_cands": {0: []}})
    sim.add(daily(_day, 3, seed=2))
    sim.add(daily(lambda d, t0, rng: _day(d, t0, rng, with_hdr=False), 3, seed=3, t0=MON + 3 * DAY))
    sim.run_until(MON + 5 * DAY + 2 * 3600)
    gone = [e.extra["attribute"] for e in sim.events("attribute_gone")]
    assert gone == ["hdr.x-client-ver"]
    reg = MP.get_registry(sim.st, "oa")
    assert reg.get("hdr.x-client-ver").state == "gone"
    assert reg.get("net.bytes_up").state == "active"


def test_dip_on_a_holiday_is_not_a_schema_change():
    sim = Sim(sel={"targets_sys": {0: []}, "split_cands": {0: []}})

    def holiday_flag(s):
        # P01 marks the previous day as not normal (a holiday)
        b = s.st.batch_at("oa", EV.EVT_BATCH, s.now)
        if b is not None:
            cb = b.aligned({}, {"day": 0, "normal_prev_day": False})
            s.st.add_batch("oa", EV.EVT_CTX, s.now, cb)
    sim.add(daily(_day, 3, seed=2))
    sim.add(daily(lambda d, t0, rng: _day(d, t0, rng, with_hdr=False), 1, seed=3, t0=MON + 3 * DAY))
    sim.add(daily(_day, 2, seed=4, t0=MON + 4 * DAY))
    sim.run_until(MON + 4 * DAY - 3600)
    sim.hooks.append(holiday_flag)
    sim.run_until(MON + 5 * DAY + 2 * 3600)
    assert not sim.events("attribute_gone")


def test_workday_only_attribute_is_not_gone_on_a_weekend():
    """An attribute that only workday actions carry (login bodies) has no
    coverage on a normal weekend: against its own day type's reference it is
    not a schema change; when it disappears on workdays it is declared gone
    within one normal workday (P01's calendar gives the previous day's class)."""
    import datetime as _dt
    from ptree_sim import TZ, is_workday, local
    sim = Sim(sel={"targets_sys": {0: []}, "split_cands": {0: []}})

    def calendar(s):
        d = local(s.now - 1.0).date()
        pc = s.st.get_model("oa", "__system__", "model.pcal") or {"days": {}, "normal": {}}
        pc["days"][d.toordinal()] = {"mass": 1.0, "class": "workday" if d.weekday() < 5 else "weekend"}
        s.st.put_model("oa", "__system__", "model.pcal", pc)
        b = s.st.batch_at("oa", EV.EVT_BATCH, s.now)
        if b is not None:
            s.st.add_batch("oa", EV.EVT_CTX, s.now,
                           b.aligned({}, {"day": d.toordinal(), "normal_prev_day": True}))
    sim.hooks.append(calendar)

    def day(d, t0, rng, body=True):
        ev = []
        for i in range(40):
            ts = t0 + rng.uniform(0, DAY)
            a = {"http.route": "GET x /home", "net.bytes_down": float(rng.lognormal(8, 0.5))}
            if is_workday(t0) and body and i % 2 == 0:
                a.update({"http.route": "POST x /login", "body.kv.username": f"u{i % 7}"})
            ev.append((ts, "oa", f"192.168.1.{i % 20}", a))
        return ev
    sim.add(daily(day, 14, seed=5))
    sim.add(daily(lambda d, t0, rng: day(d, t0, rng, body=False), 3, seed=6, t0=MON + 14 * DAY))
    sim.run_until(MON + 14 * DAY)
    assert not sim.events("attribute_gone"), [e.extra for e in sim.events("attribute_gone")]
    sim.run_until(MON + 16 * DAY + 2 * 3600)                    # Mon 14 without bodies
    gone = [e.extra["attribute"] for e in sim.events("attribute_gone")]
    assert "body.kv.username" in gone and "net.bytes_down" not in gone
