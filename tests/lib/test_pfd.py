"""lib/pfd (P08 maths, docs/lib3/progressive.md §6.12, card P08 tests)."""
from __future__ import annotations

import random

import numpy as np
import pytest

from app.engines.behavior.lib import pfd as FD
from app.engines.behavior.lib import pnode as PN

T0 = 1_780_000_000.0
DAY = 86400.0
GA = [("192.168.1.21", "jack"), ("192.168.1.23", "rose"), ("10.168.7.121", "mike")]


def _pairs(rows, t0=T0, step=DAY):
    ps = PN.PairSketch()
    for i, (x, y) in enumerate(rows):
        ps.update(x, y, t0 + i * step, 1.0, 1.0)
    return ps


def test_three_ips_fixed_usernames_lb_and_fd():
    ps = PN.PairSketch()
    for x, y in GA:
        for _ in range(5):
            ps.update(x, y, T0, 1.0, 1.0)
    r = FD.fit_pair(ps, T0)
    assert r["fd"]["holds"] and r["fd"]["one_to_one"] and r["fd"]["g3"] == 0.0
    for x, y in GA:
        e = r["table"][x]
        assert e["bound"] and e["top"] == y
        assert e["LB"] == pytest.approx(0.881, abs=0.002)        # leave-one-out prior, 5 pure logins
    ps6 = PN.PairSketch()
    for x, y in GA:
        for _ in range(6):
            ps6.update(x, y, T0, 1.0, 1.0)
    assert min(e["LB"] for e in FD.fit_pair(ps6, T0)["table"].values()) >= 0.8997


def test_jeffreys_alone_needs_more_logins():
    ps = PN.PairSketch()
    for _ in range(5):
        ps.update("192.168.9.9", "svc", T0, 1, 1)
    e = FD.fit_pair(ps, T0)["table"]["192.168.9.9"]
    assert e["LB"] < 0.8 and not e.get("bound")              # per-IP Jeffreys needs ~9


def test_cross_binding_p_and_flags():
    ps = PN.PairSketch()
    for x, y in GA:
        for _ in range(5):
            ps.update(x, y, T0, 1.0, 1.0)
    r = FD.fit_pair(ps, T0)
    p, f = FD.check_forward(r, "192.168.1.21", "rose")
    assert f == ["cross_binding"] and 0.02 < p < 0.04
    p, f = FD.check_forward(r, "192.168.1.21", "zed")
    assert f == ["unbound_value"]
    assert FD.check_forward(r, "192.168.1.21", "jack") == (1.0, [])


def test_portal_random_usernames_no_fd():
    rnd = random.Random(0)
    ps = PN.PairSketch()
    for i in range(3000):
        ip = f"10.70.{rnd.randint(0, 3)}.{rnd.randint(1, 40)}"
        ps.update(ip, f"u{rnd.randint(0, 50000)}", T0 + i * 30, 1, 1)
    r = FD.fit_pair(ps, T0 + 3000 * 30)
    assert not r["fd"]["holds"] and r["fd"]["g3"] > 0.5
    assert not any(e.get("bound") for e in r["table"].values())


def test_shared_terminal_set_binding_and_outside_flag():
    ps = PN.PairSketch()
    for i in range(40):
        ps.update("192.168.5.7", ["jack", "rose"][i % 2], T0 + i * 3600, 1, 1)
    r = FD.fit_pair(ps, T0 + 40 * 3600)
    e = r["table"]["192.168.5.7"]
    assert not e.get("bound") and e["set"] == ["jack", "rose"] and e["U"] <= 0.05
    p, f = FD.check_forward(r, "192.168.5.7", "lucy")
    assert f == ["outside_set"] and p <= 0.05


def test_service_account_reverse_set_binding():
    # reverse pair (Y -> X): the account's source hosts form a closed set
    ps = PN.PairSketch()
    for i in range(60):
        ps.update("svc_backup", ["10.1.1.5", "10.1.1.6", "10.1.1.7"][i % 3], T0 + i * 3600, 1, 1)
    r = FD.fit_pair(ps, T0 + 60 * 3600)
    e = r["table"]["svc_backup"]
    assert e["set"] == ["10.1.1.5", "10.1.1.6", "10.1.1.7"]
    p, f = FD.check_reverse(r, "10.9.9.9", "svc_backup", T0 + 61 * 3600)
    assert "foreign_source" in f and p <= 0.05
    assert FD.check_reverse(r, "10.1.1.6", "svc_backup", T0)[1] == []


def test_shared_ip_gets_no_per_ip_binding():
    ps = PN.PairSketch()
    for _ in range(30):
        ps.update("shared:192.168.5.1", "jack", T0, 1, 1)
    for x, y in GA:
        for _ in range(5):
            ps.update(x, y, T0, 1, 1)
    r = FD.fit_pair(ps, T0)
    assert "bound" not in r["table"]["shared:192.168.5.1"]
    assert all(r["table"][x]["bound"] for x, _ in GA)


def test_reverse_concurrency_and_readdress():
    ps = PN.PairSketch()          # reverse pair username -> IP
    for y, x in [(b, a) for a, b in GA]:
        for _ in range(8):
            ps.update(y, x, T0, 1, 1)
    r = FD.fit_pair(ps, T0)
    now = T0 + 10 * DAY
    act = {"10.168.7.121": now - 600}
    p, f = FD.check_reverse(r, "10.168.99.5", "mike", now, act.get, x_known=False,
                            day_start=now - 9 * 3600)
    assert f == ["foreign_source", "concurrent_use"]
    act = {"10.168.7.121": now - 2 * DAY}
    p, f = FD.check_reverse(r, "10.168.99.5", "mike", now, act.get, x_known=False,
                            day_start=now - 9 * 3600)
    assert f == ["foreign_source", "readdress_candidate"]
    act = {"10.168.7.121": now - 5 * 3600}
    p, f = FD.check_reverse(r, "10.168.99.5", "mike", now, act.get, x_known=False,
                            day_start=now - 9 * 3600)
    assert f == ["foreign_source"]


def test_rebinding_segment_restart():
    ps = PN.PairSketch()
    for d in range(20):
        for x, y in GA:
            ps.update(x, y, T0 + d * DAY, 1, 1)
    tr = FD.RebindTracker()
    t = T0 + 20 * DAY
    for d in range(3):
        for k in range(2):
            ts = t + d * DAY + k * 60
            ps.update("10.168.7.121", "mike.w", ts, 1, 1)
            tr.observe("10.168.7.121", "mike.w", ts, 800000 + d, "mike", True)
    now = t + 3 * DAY
    assert tr.rebind_candidate("10.168.7.121", "mike") == "mike.w"
    seg = {"10.168.7.121": FD.rebind_baseline(ps, "10.168.7.121", "mike.w", now)}
    r = FD.fit_pair(ps, now, seg)
    e = r["table"]["10.168.7.121"]
    assert e["top"] == "mike.w" and e["bound"] and e["n"] == pytest.approx(e["k"])
    # without the segment restart the old value still dominates
    assert FD.fit_pair(ps, now)["table"]["10.168.7.121"]["top"] == "mike"


def test_rebinding_needs_two_normal_days_and_absent_old_value():
    tr = FD.RebindTracker()
    for k in range(6):
        tr.observe("x", "new", T0 + k * 60, 800000, "old", True)        # one day only
    assert tr.rebind_candidate("x", "old") is None
    tr.observe("x", "new", T0 + DAY, 800001, "old", False)             # not a normal day
    assert tr.rebind_candidate("x", "old") is None
    tr.observe("x", "new", T0 + 2 * DAY, 800002, "old", True)
    assert tr.rebind_candidate("x", "old") == "new"
    tr.observe("x", "old", T0 + 2 * DAY + 60, 800002, "old", True)    # old value still used
    assert tr.rebind_candidate("x", "old") is None


def test_screening_finds_ip_username_and_rejects_portal():
    rnd = random.Random(1)
    rows, w = [], []
    for d in range(10):
        for x, y in GA:
            rows.append({"net.src": x, "body.kv.username": y, "body.kv.captcha": str(rnd.randint(1000, 9999))})
            w.append(1.0)
        for _ in range(30):
            rows.append({"net.src": f"10.70.0.{rnd.randint(1, 250)}",
                         "q.kv.user": f"u{rnd.randint(0, 99999)}"})
            w.append(1.0)
    gen = lambda a, l, v: v if l == 0 else ".".join(v.split(".")[:3]) + ".0/24"   # noqa: E731
    specs = FD.screen(rows, np.asarray(w), [("net.src", 0), ("net.src", 1)],
                      ["body.kv.username", "body.kv.captcha", "q.kv.user"], gen)
    got = {(d["x"], d["y"], d["dir"]) for d in specs}
    assert ("net.src", "body.kv.username", "fwd") in got
    assert ("body.kv.username", "net.src", "rev") in got
    assert not any("q.kv.user" in (d["x"], d["y"]) for d in specs)
    assert not any("captcha" in d["y"] or "captcha" in d["x"] for d in specs)


def test_probe_keeps_rare_stratum():
    pr = FD.ProbeReservoir(r_k=32, s_max=8)
    rnd = random.Random(2)
    for i in range(20000):
        pr.offer("portal", {"net.src": f"1.1.{i % 200}.1", "q.kv.a": str(i)}, 1.0, T0 + i, rnd.random())
        if i % 1000 == 0:
            pr.offer("login", {"net.src": "192.168.1.21", "body.kv.username": "jack"}, 1.0, T0 + i,
                     rnd.random())
    rows, w = pr.rows(T0 + 20000)
    assert sum(1 for r in rows if r.get("body.kv.username")) == 20
    assert pr.n_rows() <= 8 * 32
    # HT weights restore the stratum masses
    dec = sum(2.0 ** (-(20000 - i) / (7 * DAY)) for i in range(20000)) + \
        sum(2.0 ** (-(20000 - i) / (7 * DAY)) for i in range(0, 20000, 1000))
    assert w.sum() == pytest.approx(dec, rel=1e-6)
