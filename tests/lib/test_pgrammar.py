"""lib/pgrammar (P07 maths, docs/lib3/progressive.md §6.11, card P07 tests)."""
from __future__ import annotations

import random
import re

import pytest

from app.engines.behavior.lib import pgrammar as G
from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib.phier import Shaped, shape

T0 = 1_780_000_000.0
DAY = 86400.0


def _text(values, days=20, t0=T0, policy="clear"):
    ts = PN.TextSummary(policy)
    for d in range(days):
        for i, v in enumerate(values):
            ts.update(v, t0 + d * DAY + i * 60, 1.0, 1.0)
    return ts, t0 + days * DAY


def test_three_usernames_give_a_z4_and_closed_set():
    ts, t = _text(["jack", "rose", "mike"], days=30)          # ~66 units on the H_l channel
    r = G.fit_text(ts, t)
    assert r["grammar"] == "[a-z]{4}"
    assert r["len"] == [4, 4] and r["c_g"] == pytest.approx(1.0)
    assert r["closed"] == ["jack", "mike", "rose"]
    assert r["U"] <= 0.01
    # the grammar is learned, not pre-set: never {1,10}
    assert "10" not in r["grammar"]


def test_closed_set_needs_evidence():
    ts, t = _text(["jack", "rose", "mike"], days=5)           # 15 units < n_conf = 20
    r = G.fit_text(ts, t)
    assert r["grammar"] == "[a-z]{4}" and "closed" not in r


def test_rename_mike_w_prefix_factoring():
    ts, t = _text(["jack", "rose", "mike"], days=20)
    for i in range(6):
        ts.update("mike.w", t + i * 3600, 1, 1)
    r = G.fit_text(ts, t + 7 * 3600)
    assert r["grammar"] == r"[a-z]{4}(\.[a-z])?"
    rx = re.compile(r["grammar"])
    assert all(rx.fullmatch(v) for v in ("jack", "rose", "mike.w"))
    assert not rx.fullmatch("mike.ww")


def test_anti_unification_spec_example():
    assert G._build([[("L", 4, 4)], [("L", 3, 3)], [("L", 5, 5), (".", 1, 1), ("L", 1, 1)]],
                    "a-z") == r"[a-z]{3,5}(\.[a-z])?"


def test_lengths_3_to_8_learned_not_preset_and_pin_widens():
    rnd = random.Random(1)
    vals = ["".join(rnd.choice("abcdefghijklmnop") for _ in range(rnd.randint(3, 8)))
            for _ in range(300)]
    ts, t = _text(vals, days=1)
    r = G.fit_text(ts, t)
    assert r["grammar"] == "[a-z]{3,8}"
    assert r["len"] == [3, 8]
    p = G.fit_text(ts, t, pin={"len_max": 10})
    assert p["grammar"] == "[a-z]{3,10}" and p["len"] == [3, 10]
    q = G.fit_text(ts, t, pin={"len_max": 5})              # pins never narrow
    assert q["grammar"] == "[a-z]{3,8}"


def test_injection_shape_flag_and_p_values():
    ts, t = _text(["jack", "rose", "mike"], days=30)
    r = G.fit_text(ts, t)
    p, flags = G.check_text(r, "admin' OR '1'='1")
    assert "injection_shape" in flags and "grammar" in flags
    assert p <= r["U_s"] + 1e-12 and p < 0.05
    p2, f2 = G.check_text(r, "jack")
    assert p2 == 1.0 and not f2
    p3, f3 = G.check_text(r, "lucy")                          # grammar ok, not in the closed set
    assert f3 == ["new_value"] and p3 == pytest.approx(r["U"])
    p4, f4 = G.check_text(r, "jacky")
    assert "length" in f4 and "injection_shape" not in f4


def test_fragmented_hex_falls_back_to_charset_union():
    rnd = random.Random(2)
    vals = []
    for _ in range(400):
        n = rnd.randint(8, 16)
        vals.append(Shaped(shape("".join(rnd.choice("0123456789abcdef") for _ in range(n)))))
    ts = PN.TextSummary("shape")
    for i, v in enumerate(vals):
        ts.update(v, T0 + i * 60, 1, 1)
    r = G.fit_text(ts, T0 + 400 * 60)
    assert r["mode"] in ("charset", "shape")
    rx = re.compile(r["grammar"])
    for v in ("0a1b2c3d", "deadbeef00112233"):
        assert rx.fullmatch(v), (r["grammar"], v)
    assert "closed" not in r                                   # shape-only policy has no values


def test_required_keys_and_new_key_p():
    ss = PN.SetSummary()
    keys = frozenset({"username", "password", "captcha", "csrf", "viewstate"})
    for d in range(30):
        ss.update(keys, T0 + d * DAY, 1, 1)
    ss.update(keys | {"debug"}, T0 + 30 * DAY, 1, 1)
    r = G.fit_set(ss, T0 + 30 * DAY)
    assert r["required"] == sorted(keys)
    assert r["optional"] == ["debug"]
    p, f = G.check_set(r, {"username", "password", "captcha", "csrf", "viewstate", "cmd"})
    assert f == ["new_key"] and p == pytest.approx(r["p_new_key"]) and p < 0.1
    p, f = G.check_set(r, {"username"})
    assert "missing_key" in f and p < 0.05


def test_categorical_closed_set():
    cs = PN.CatSummary()
    for d in range(60):
        cs.update(["同意", "驳回", "退回"][d % 3], T0 + d * 3600, 1, 1)
    r = G.fit_cat(cs, T0 + 61 * 3600)
    assert r["closed"] == sorted(["同意", "驳回", "退回"])
    assert G.check_cat(r, "同意") == (1.0, [])
    p, f = G.check_cat(r, "批准")
    assert f == ["new_value"] and p <= 0.01


def test_shape_gain_positive_for_concentrated_node():
    reg_top = PN.CatSummary()
    rnd = random.Random(3)
    for i in range(2000):
        reg_top.update(shape("".join("a" * rnd.randint(1, 12))), T0 + i, 1, 1)
    ts, t = _text(["jack", "rose", "mike"], days=5)
    assert G.shape_gain(ts.shapes, reg_top.ss, t) > 1.5


def _hex(rnd):
    return "".join(rnd.choice("0123456789abcdef") for _ in range(rnd.randint(8, 16)))


def test_grammar_confidence_is_calibrated_and_rises_with_evidence():
    """Shape-only hex tokens (passwords) and free text: the stated confidence
    c_g (1 - U_s) is met by fresh values in (nearly) every fit, and it rises
    with the evidence (the length range carries the 2/(n+1) rank bound; a
    charset grammar's coverage comes from the histograms, not from the tracked
    shapes, whose guaranteed mass shrinks under churn)."""
    rnd = random.Random(3)
    means = []
    for n in (20, 80, 160):
        viol, confs = 0, []
        for rep in range(25):
            ts = PN.TextSummary("shape")
            for i in range(n):
                ts.update(Shaped(shape(_hex(rnd))), T0 + i * 3600, 1.0, 1.0)
            r = G.fit_text(ts, T0 + n * 3600)
            rx = re.compile(r["grammar"])
            hold = sum(bool(rx.fullmatch(_hex(rnd))) for _ in range(300)) / 300
            conf = r["c_g"] * (1 - r["U_s"])
            viol += hold < conf - 0.05
            confs.append(conf)
            assert r["confidence"] == pytest.approx(conf)
        assert viol <= 2, (n, viol)
        means.append(sum(confs) / len(confs))
    assert means[0] < means[1] < means[2] and means[2] >= 0.97, means


def test_short_mixed_tokens_fold_into_the_alnum_class():
    """8-character hex tokens are kept in detail by the shape (runs <= 8), 9-16
    character ones collapse to A<n>: anti-unification over the class hierarchy
    puts both under one skeleton, [A-Za-z0-9]{8,16}."""
    assert G._alnum_fold("D1 L2 D1 L1 D3") == "A8" and G._alnum_fold("L4") == "L4"
    assert G._alnum_fold("L4 . L1") == "L4 . L1"
    rnd = random.Random(5)
    ts = PN.TextSummary("shape")
    for d in range(12):
        for k in range(3):
            v = "".join(rnd.choice("0123456789abcdef") for _ in range(8 if k == 0 else 12))
            v = v if len(set(v) & set("abcdef")) and len(set(v) & set("0123456789")) else "3fa2c9d1"
            ts.update(Shaped(shape(v)), T0 + d * DAY + k * 600, 1.0, 1.0)
    r = G.fit_text(ts, T0 + 12 * DAY)
    rx = re.compile(r["grammar"])
    assert rx.fullmatch("3fa2c9d1") and rx.fullmatch("0a1b2c3d4e5f") and not rx.fullmatch("3fa2-9d1")
    # low churn among the tracked shapes: one skeleton, the exact length range
    ts2 = PN.TextSummary("shape")
    for d in range(12):
        for v in ("3fa2c9d1", "0a1b2c3d4e5f", "9e8d7c6b5a4f"):
            ts2.update(Shaped(shape(v)), T0 + d * DAY, 1.0, 1.0)
    r2 = G.fit_text(ts2, T0 + 12 * DAY)
    assert r2["grammar"] == "[A-Za-z0-9]{8,12}" and r2["mode"] == "shape"


def test_closed_set_of_mixed_type_values():
    """A categorical target may hold numbers and the absence value together
    (found on the real lattice: status codes next to '⊥')."""
    cs = PN.CatSummary()
    for d in range(30):
        for v in (200, 302, "⊥"):
            cs.update(v, T0 + d * DAY, 1.0, 1.0)
    r = G.fit_cat(cs, T0 + 30 * DAY)
    assert set(map(str, r["closed"])) == {"200", "302", "⊥"}
    assert G.check_cat(r, 404)[1] == ["new_value"] and G.check_cat(r, 302)[0] == 1.0
