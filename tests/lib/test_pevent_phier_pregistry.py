"""lib/pevent.py (event batch, sampling), lib/phier.py (generalisation
hierarchies) and lib/pregistry.py (schema inference, statistics, hierarchy
models, schema change)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import phier as H
from app.engines.behavior.lib import pregistry as RG
from app.engines.behavior.lib import psketch as PS

DAY = PS.DAY


# ------------------------------------------------------------------- pevent
def _batch():
    b = EV.BatchBuilder("oa")
    b.add(10.0, "192.168.1.21", {"http.route": "POST oa /login", "body.kv.username": "jack",
                                 "net.bytes_up": 1500})
    b.add(11.0, "192.168.1.23", {"http.route": "POST oa /login", "body.kv.username": "rose",
                                 "net.bytes_up": 1300})
    b.add(12.0, "192.168.1.21", {"http.route": "GET oa /approval/list", "net.bytes_up": 300},
          w=4.0, flags=EV.FLAG_APPROX)
    return b.build(0.0, 60.0)


def test_batch_columnar_access_absence_and_select():
    bt = _batch()
    assert bt.n == 3 and bt.ips == ["192.168.1.21", "192.168.1.23"]
    assert bt.cols["net.bytes_up"].numeric and not bt.cols["http.route"].numeric
    assert bt.get("body.kv.username", 1) == "rose"
    assert bt.get("body.kv.username", 2) is EV.ABSENT          # absence is a value
    assert list(bt.dense("net.bytes_up", 0.0)) == [1500.0, 1300.0, 300.0]
    assert bt.row(2) == {"http.route": "GET oa /approval/list", "net.bytes_up": 300.0}
    sub = bt.select([0, 2], ["net.bytes_up"])
    assert sub.n == 2 and list(sub.rid) == [0, 2] and sub.ips == ["192.168.1.21"]
    assert sub.names() == ["net.bytes_up"] and sub.get("net.bytes_up", 1) == 300.0
    assert sub.event_id(1) == ("oa", 0, 60.0, 2)                 # ids survive compaction
    ctx = bt.aligned(EV.cols_from_rows(3, [{"ctx.tod_min": 540}, {}, {"ctx.tod_min": 541}]))
    assert ctx.get("ctx.tod_min", 2) == 541 and ctx.get("ctx.tod_min", 1) is EV.ABSENT


def test_compact_keeps_learned_and_held_rows():
    bt = _batch()
    bt.learn[:] = [True, False, False]
    c = EV.compact_batch(bt, ["http.route"], extra_rows=[2])
    assert list(c.rid) == [0, 2] and c.names() == ["http.route"]
    assert c.nbytes() < bt.nbytes()


def test_attr_name_and_flatten():
    assert EV.attr_name("Body", "KV", "User Name") == "body.kv.user_name"
    long = EV.attr_name("hdr", "x" * 200)
    assert len(long) <= EV.NAME_MAX and "~" in long
    out = {}
    EV.flatten("body.kv", {"items": [{"name": "a", "q": 1}, {"name": "b"}],
                           "tags": ["x", "y"], "user": {"id": 7, "ok": True}}, out)
    assert out["body.kv.items[].name"] == frozenset({"a", "b"})
    assert out["body.kv.items[].n"] == 2
    assert out["body.kv.tags[]"] == frozenset({"x", "y"})
    assert out["body.kv.user.id"] == 7 and out["body.kv.user.ok"] == 1
    assert EV.template_key("items[3].name") == "items[].name"


@pytest.mark.parametrize("seed", range(3))
def test_threshold_sampling_rare_strata_full_and_ht_unbiased(seed):
    """(e) of P00: every event of a stratum with c_k <= tau is kept, and
    HT-weighted totals match the truth within 1 % over 100 seeds."""
    rng = np.random.default_rng(seed)
    strata = ["portal"] * 5000 + ["login"] * 40 + ["approve"] * 3
    counts = {"portal": 5000, "login": 40, "approve": 3}
    tau = EV.threshold_tau(list(counts.values()), 1000)
    assert sum(min(c, tau) for c in counts.values()) == pytest.approx(1000)
    ests = []
    for s in range(100):
        b = EV.BatchBuilder("s")
        for k in strata:
            b.add(0.0, "ip", {"k": k})
        bt = b.build(0, 1)
        u = rng.random(bt.n)
        EV.select_learning_sample(bt, strata, 1000, u)
        rare = [i for i, k in enumerate(strata) if k != "portal"]
        assert bt.learn[rare].all() and (bt.pi[rare] == 1).all()
        ests.append(bt.mass().sum())
    assert np.mean(ests) == pytest.approx(len(strata), rel=0.01)


def test_priority_sampling_unbiased():
    rng = np.random.default_rng(0)
    w = np.r_[np.full(900, 1.0), np.full(100, 10.0)]
    tots = []
    for _ in range(300):
        idx, ht = EV.priority_sample(w, 100, rng.random(w.size))
        assert len(idx) == 100
        tots.append(ht.sum())
    assert np.mean(tots) == pytest.approx(w.sum(), rel=0.03)
    idx, ht = EV.priority_sample([1, 2], 5, [0.5, 0.5])
    assert list(ht) == [1, 2]


def test_pconfig_merges_defaults():
    cfg = EV.pconfig({"progressive": {"enabled": True, "defaults": {"e_rate": 3}}})
    assert cfg["enabled"] and cfg["defaults"]["e_rate"] == 3
    assert cfg["defaults"]["w_max"] == 512 and "secret_globs" in cfg["value_policy"]
    assert EV.learn_delay_s(60.0) == 600.0 and EV.learn_delay_s(900.0) == 3600.0


# -------------------------------------------------------------------- phier
def test_ip_hierarchy_levels_groups_regions_shared():
    reg = H.Regions([("GA-net", ["192.168.1.0/24"])])
    h = H.Hierarchies({"net.src": {"type": "ip"}}, ip2g={"192.168.1.21": "GA"}, regions=reg,
                      shared={"10.0.0.9"})
    a = "net.src"
    assert h.levels(a) == ["ip", "/24", "/16", "grp", "reg", "*"]
    v = "192.168.1.21"
    assert [h.gen(a, l, v) for l in range(6)] == [
        v, "192.168.1.0/24", "192.168.0.0/16", "grp:GA", "reg:GA-net", "*"]
    assert h.gen(a, 3, "192.168.1.99") == H.GRP_NONE
    assert h.gen(a, 0, "10.0.0.9") == "shared:10.0.0.9"
    assert h.gen(a, 1, "2001:db8:1:2::5") == "2001:db8:1:2::/64"
    assert h.gen(a, 2, EV.ABSENT) is EV.ABSENT
    h2 = H.Hierarchies({"net.src": {"type": "ip"}})              # no group / region model
    assert h2.gen(a, 3, v) == "*" and h2.gen(a, 4, v) == "*"      # falls through
    assert h.card_hint(a, 0) == 2 ** 32 and h.card_hint(a, 5) == 1


def test_route_time_status_text_set_levels():
    h = H.Hierarchies({"http.status": {"type": "categorical"},
                       "body.kv.username": {"type": "text"},
                       "body.keys": {"type": "set", "hier": {"set_keep": {"username", "password"}}}},
                      windows=[(540, 561, "w:0900-0921")],
                      windows_dt={"wd": [(540, 561, "w:0900-0921")]})
    r = "POST oa.local /approval/{num}/approve"
    assert [h.gen("http.route", l, r) for l in range(5)] == [
        r, "oa.local /approval/{num}/*", "oa.local /approval/*", "oa.local", "*"]
    assert h.gen("ctx.tod_min", 1, 545) == 36
    assert h.gen("ctx.tod_min", 2, 545) == "w:0900-0921"
    assert h.gen("ctx.tod_min", 2, 900) == "w:off"
    assert h.gen("ctx.tod_min", 3, 545) == "day"
    assert h.gen("ctx.when", 2, ("workday", 545)) == ("wd", "w:0900-0921")
    assert h.gen("ctx.when", 3, ("nonworkday", 100)) == "nwd_night"
    assert h.gen("http.status", 1, 302) == "3xx"
    assert [h.gen("body.kv.username", l, "mike.w") for l in range(6)] == [
        "mike.w", "L4 . L1", "L4-7 . L1", "{.,L}:4-7", "len:4-7", "*"]
    assert h.gen("body.keys", 1, frozenset({"username", "password", "rare"})) == \
        frozenset({"username", "password"})
    assert h.gen("body.keys", 2, frozenset({"a", "b"})) == "card:2-3"


def test_shapes_and_skeletons():
    assert H.shape("jack") == "L4"
    assert H.shape("mike.w") == "L4 . L1"
    assert H.shape("3fa2c9") == "D1 L2 D1 L1 D1"
    assert H.shape("aB3dE5fG7hJ") == "A11"
    assert H.shape("admin' OR '1'='1").startswith("L5 '")
    assert H.skeleton(H.shape("mike.w")) == "L . L"
    assert H.shape_length("L4 . L1") == 6


# ---------------------------------------------------------------- pregistry
def _feed(reg, name, values, t, kind=0):
    reg.observe_events(kind, t, len(values))
    reg.observe(name, values, t, 1.0, None, None, kind)


def test_type_inference_on_synthetic_columns():
    rng = np.random.default_rng(0)
    reg = RG.AttrRegistry("s")
    t = 0.0
    cols = {
        "net.src": [f"10.0.{i % 7}.{i % 200}" for i in range(600)],
        "net.bytes_up": list(rng.lognormal(7, 1, 600)),
        "http.status": list(rng.choice([200, 302, 404], 600)),
        "ctx.sess_pos": list(rng.integers(0, 6, 600)),
        "body.keys": [frozenset({"username", "password"})] * 600,
        "hdr.x-client-ver": list(rng.choice(["1.2", "1.3", "2.0beta"], 600)),
        "body.tpl": [f"free text comment number {i} " + "x" * (i % 13) for i in range(600)],
        "hdr.x-prefs": [f"a={i}&b={i % 5}" for i in range(600)],
        "meta.login_ts": list(1.7e9 + rng.random(600) * 1e6),
    }
    for nm, vals in cols.items():
        _feed(reg, nm, vals, t)
    reg.update_types(t)
    typ = {n: reg.get(n).type for n in cols}
    assert typ["net.src"] == "ip"
    assert typ["net.bytes_up"] == "numeric"
    assert typ["http.status"] == "categorical"                 # code hint
    assert typ["ctx.sess_pos"] == "ordinal"
    assert typ["body.keys"] == "set"
    assert typ["hdr.x-client-ver"] == "categorical"
    assert typ["body.tpl"] == "text"
    assert typ["hdr.x-prefs"] == "text" and reg.get("hdr.x-prefs").parse_as == "form"
    assert typ["meta.login_ts"] == "time"
    assert not reg.get("net.src").locked
    reg.update_types(t + DAY)                                    # n >= 500 and >= 1 day
    assert reg.get("net.src").locked
    reg.update_stats(t)
    assert reg.get("net.bytes_up").hier["log"] is True


def test_numeric_bins_refresh_only_above_jsd_threshold():
    rng = np.random.default_rng(1)
    reg = RG.AttrRegistry("s")
    _feed(reg, "net.bytes_up", list(rng.lognormal(7, 0.5, 2000)), 0.0)
    reg.update_types(0.0)
    reg.update_stats(0.0)
    assert reg.refresh_hierarchies(0.0) == ["net.bytes_up"]
    edges = reg.get("net.bytes_up").hier["edges"]
    assert len(edges) == 7
    h = H.Hierarchies(reg)
    bins = [h.gen("net.bytes_up", 1, x) for x in rng.lognormal(7, 0.5, 4000)]
    occ = np.bincount(bins, minlength=8) / 4000
    assert np.all(np.abs(occ - 0.125) < 0.03)
    _feed(reg, "net.bytes_up", list(rng.lognormal(7, 0.5, 2000)), DAY)
    assert reg.refresh_hierarchies(2 * DAY) == []                # same distribution: kept
    _feed(reg, "net.bytes_up", list(rng.lognormal(9, 0.5, 20000)), 3 * DAY)
    assert reg.refresh_hierarchies(4 * DAY) == ["net.bytes_up"]  # moved: refreshed
    assert reg.refresh_hierarchies(4.5 * DAY) == []              # at most daily


def test_cap_and_overflow_admission():
    reg = RG.AttrRegistry("s", a_max=4)
    for i in range(4):
        _feed(reg, f"hdr.h{i}", ["v"], 0.0)
    assert reg.register("hdr.new", 0.0) == "overflow" and reg.overflow_n == 1
    reg.set_role("hdr.h2", "dropped")
    assert reg.register("hdr.new", 1.0) == "new"
    assert "hdr.h2" not in reg and len(reg) == 4
    assert reg.register("hdr.late", 31 * DAY) == "new"          # 30-d unseen evicted


def test_gone_on_normal_day_not_on_holiday_and_revival():
    """A header that stops arriving at normal volume is declared gone at the
    end of the first full normal day; a day that is a holiday never counts."""
    reg = RG.AttrRegistry("s", day_offset_s=0.0)
    for d in range(20):
        for h in range(0, 24, 2):
            t = d * DAY + h * 3600
            reg.observe_events(0, t, 10)
            reg.observe("hdr.x-ver", ["1"] * 10, t)
            reg.observe("http.method", ["GET"] * 10, t)
    t0 = 20 * DAY
    gone_at = None
    for h in range(0, 24 * 4, 2):
        t = t0 + h * 3600
        reg.observe_events(0, t, 10)
        reg.observe("http.method", ["GET"] * 10, t)
        holiday_prev = (t - t0) < 2 * DAY + 1       # days 20 and 21 are holidays
        g = reg.check_gone(t, normal_day=not holiday_prev)
        if g:
            gone_at = t
            assert g == ["hdr.x-ver"]
    assert gone_at == t0 + 3 * DAY                   # end of day 22, the first normal day
    assert reg.get("http.method").state == "active"
    assert reg.register("hdr.x-ver", gone_at + 3600) == "revived"


def test_shaped_values_mix_with_clear_values_of_one_attribute():
    """A value the policy shaped (long / secret-like) and a clear value of the
    same attribute share one hierarchy: the shaped one is its own shape."""
    from app.engines.behavior.lib import pnode as PN
    from app.engines.behavior.lib import pparse as PP
    pol = PP.ValuePolicy(EV.pconfig({})["value_policy"])
    long_v, mode = pol.apply("body.kv.comment", "word " * 20)
    assert mode == "shape" and isinstance(long_v, H.Shaped)
    assert H.shape(long_v) is long_v
    h = H.Hierarchies({"body.kv.comment": {"type": "text"}})
    assert h.gen("body.kv.comment", 1, long_v) == long_v
    assert h.gen("body.kv.comment", 4, long_v) == "len:32+"
    assert h.gen("body.kv.comment", 1, "short note") == "L5 SP L4"
    ts = PN.TextSummary()
    ts.update(long_v, 0.0, 1.0, 1.0)
    ts.update("short note", 0.0, 1.0, 1.0)
    assert {k for k, *_ in ts.shapes.items(0.0)} == {long_v, "L5 SP L4"}
    assert [k for k, *_ in ts.values.items(0.0)] == ["short note"]    # shaped: no exact value
    reg = RG.AttrRegistry("s")
    reg.observe("body.kv.comment", [long_v, "short note"], 0.0)
    reg.get("body.kv.comment").type = "text"
    assert reg.get("body.kv.comment").key(long_v) == long_v


def test_hex_digests_are_not_numbers_and_tuples_stay_tuples():
    reg = RG.AttrRegistry("s")
    vals = [f"{i}e{150 + i % 50}ab" if i % 2 else f"5e15{i % 10}" for i in range(40)]
    reg.observe("sess.key", vals, 0.0)                   # would overflow as floats
    assert reg.get("sess.key").num is None
    reg.observe("ctx.when", [("wd", 540), ("wd", 541)] * 10, 0.0)
    reg.update_types(0.0)
    r = reg.get("ctx.when")
    assert r.type == "categorical" and r.key(("wd", 540)) == ("wd", 540)
