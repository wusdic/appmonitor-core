"""lib/pselect (P05 maths, docs/lib3/progressive.md §6.4): stratified probe,
attribute evaluation, roles with hysteresis, 'IP is not a feature', rotation."""
from __future__ import annotations

import numpy as np
import pytest

from app.engines.behavior.lib import pselect as SEL
from app.engines.behavior.lib.phier import Hierarchies

T0 = 1_780_000_000.0
ROUTES = [f"GET h /r{i}" for i in range(8)]


def _hier(types):
    return Hierarchies(registry={a: {"type": t} for a, t in types.items()})


def _synthetic(n_rows=2048, n_inf=60, n_noise=200, n_const=40, seed=0, random_ip=False):
    """Rows of 300 attributes: informative ones depend on the route (the context
    seed), noise ones do not, constants never change; plus a redundant copy of
    informative attribute 0 and a department-like net.src."""
    rng = np.random.default_rng(seed)
    rows = []
    # per-route distributions of the informative attributes (4 values, peaked)
    peak = rng.integers(0, 4, size=(n_inf, len(ROUTES)))
    ips_dept = [f"192.168.{r % 4}.{10 + r}" for r in range(len(ROUTES))]
    for i in range(n_rows):
        r = int(rng.integers(0, len(ROUTES)))
        row = {"http.route": ROUTES[r], "ev.ch": "http"}
        row["net.src"] = (f"10.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}"
                          if random_ip else ips_dept[r])
        for j in range(n_inf):
            row[f"inf{j:02d}"] = f"v{peak[j, r]}" if rng.random() < 0.9 else f"v{rng.integers(0, 4)}"
        row["dup00"] = "d" + row["inf00"]
        for j in range(n_noise):
            row[f"noise{j:03d}"] = f"n{rng.integers(0, 4)}"
        for j in range(n_const):
            row[f"const{j:02d}"] = "c"
        rows.append(row)
    types = {a: "categorical" for a in rows[0]}
    types["net.src"] = "ip"
    return rows, types


def _probe(rows, R=4096):
    pr = SEL.StratifiedProbe(R)
    for i, row in enumerate(rows):
        pr.offer(("http", row["http.route"]), row, 1.0, T0 + i, (i * 0.61803398875) % 1.0 + 1e-9)
    pr.rebalance(T0 + len(rows))
    return pr


@pytest.fixture(scope="module")
def synth_eval():
    rows, types = _synthetic()
    pr = _probe(rows)
    hier = _hier(types)
    t = T0 + len(rows)
    names = [a for a in types if a not in ("http.route", "ev.ch")]
    targets = [f"inf{j:02d}" for j in range(16)]
    st = SEL.evaluate(pr, t, hier, names, targets_prev=targets, splits_prev=["http.route"])
    kept = [a for a, v in st.items() if SEL.targetable(a) and v["U_t"] >= SEL.U_HI]
    red = SEL.redundancy(pr, t, hier, sorted(kept), cost=lambda a: 0.0, cov=lambda a: 1.0)
    ipinfo = SEL.ip_information(pr, t, hier, targets)
    out = SEL.assign_roles(st, {}, red, ipinfo, t, lambda a: (0,))
    return st, out, types


def test_roles_on_300_attributes(synth_eval):
    st, out, types = synth_eval
    roles = out["roles"]
    inf = [a for a in types if a.startswith("inf")]
    noise = [a for a in types if a.startswith("noise")]
    const = [a for a in types if a.startswith("const")]
    kept_inf = sum(roles[a] in ("target", "split", "redundant") for a in inf) / len(inf)
    dropped_noise = sum(roles[a] in ("dropped", "shape") for a in noise) / len(noise)
    inv = sum(roles[a] == "invariant" for a in const) / len(const)
    assert kept_inf >= 0.90, kept_inf
    assert dropped_noise >= 0.95, dropped_noise
    assert inv == 1.0
    # the redundant copy of an informative attribute (a 1-1 recoding)
    assert "dup00" in out["redundant"] or "inf00" in out["redundant"]
    assert roles.get("dup00") == "redundant" or roles.get("inf00") == "redundant"
    # noise is never a split candidate; the department-like IP is
    split_attrs = {a for a, _ in out["split_cands"][0]}
    assert not any(a.startswith("noise") for a in split_attrs)
    assert "net.src" in split_attrs
    assert out["who_mode"] != "none"
    tg = out["targets_sys"][0]
    assert tg and all(a.startswith(("inf", "dup")) for a in tg)


def test_net_src_dropped_on_random_ip_population():
    rows, types = _synthetic(n_rows=2048, n_inf=8, n_noise=4, n_const=2, random_ip=True)
    pr = _probe(rows)
    hier = _hier(types)
    t = T0 + len(rows)
    targets = [f"inf{j:02d}" for j in range(8)]
    names = [a for a in types if a not in ("http.route", "ev.ch")]
    st = SEL.evaluate(pr, t, hier, names, targets_prev=targets, splits_prev=["http.route"])
    ipinfo = SEL.ip_information(pr, t, hier, targets)
    assert max(ipinfo.values()) < SEL.IP_INFO_MIN
    prev = {"roles": {"net.src": "split"}, "levels": {"net.src": [0]}}
    out = SEL.assign_roles(st, prev, {}, ipinfo, t, lambda a: (0,))
    assert out["who_mode"] == "none"
    assert "net.src" not in {a for a, _ in out["split_cands"][0]}


def test_stratified_probe_keeps_rare_stratum():
    pr = SEL.StratifiedProbe(1024)
    k = 0
    for i in range(50_000):
        pr.offer(("http", "portal"), {"a": i % 7}, 1.0, T0 + i, ((i * 0.754877666) % 1.0) + 1e-9)
        if i % 1000 == 0:
            for _ in range(1):
                pr.offer(("http", "login"), {"a": 99}, 1.0, T0 + i, ((k * 0.5698402909) % 1.0) + 1e-9)
                k += 1
    pr.rebalance(T0 + 50_000)
    rows, w, strata = pr.rows(T0 + 50_000)
    n_rare = sum(1 for s in strata if s[1] == "login")
    assert n_rare >= min(32, k)
    assert len(rows) <= 1024 + 32 * 2
    # probe weights keep the population share (mass-consistent estimates)
    share = w[[s[1] == "login" for s in strata]].sum() / w.sum()
    assert share == pytest.approx(k / (50_000 + k), rel=0.2)


def test_rotation_evaluates_every_attribute_daily():
    names = [f"a{i:03d}" for i in range(500)]
    roles = {n: ("target" if i < 20 else "probe") for i, n in enumerate(names)}
    seen = set()
    for run in range(24):
        todo = SEL.names_to_evaluate(names, roles, {}, T0, run)
        assert len(todo) <= 20 + SEL.A_PROBE
        assert all(n in todo for n in names[:20])
        seen.update(todo)
    assert seen == set(names)


def test_hysteresis_demotes_after_three_low_runs():
    st = {"x": {"H": 1.0, "cov": 1.0, "U_t": 0.01, "U_s_max": -1.0, "best_levels": [], "S": 1.0,
                "distinct0": 0.01, "CR0": 0.5, "level": 0, "CR": 0.0}}
    prev = {"roles": {"x": "target"}}
    roles = []
    for _ in range(3):
        prev = SEL.assign_roles(st, prev, {}, {}, T0, lambda a: (0,))
        roles.append(prev["roles"]["x"])
    assert roles == ["target", "target", "dropped"]
    # promotion needs u_hi; a demoted attribute at 0.03 stays dropped
    st["x"]["U_t"] = 0.03
    assert SEL.assign_roles(st, prev, {}, {}, T0, lambda a: (0,))["roles"]["x"] == "dropped"
    st["x"]["U_t"] = 0.06
    assert SEL.assign_roles(st, prev, {}, {}, T0, lambda a: (0,))["roles"]["x"] == "target"


def test_same_source_and_targetable():
    assert SEL.same_source("http.route", "http.method")
    assert SEL.same_source("net.src", "net.peer_src")
    assert SEL.same_source("body.keys", "body.kv.username")
    assert not SEL.same_source("net.src", "body.kv.username")
    assert not SEL.targetable("net.src") and not SEL.targetable("ctx.sid")
    assert not SEL.targetable("ev.ch") and SEL.targetable("body.kv.username")


def test_value_groups_merge_equivalent_values():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(3000):
        v = f"c{rng.integers(0, 6)}"
        # c0..c2 behave the same way, c3..c5 another way
        tgt = ("x" if rng.random() < 0.9 else "y") if v in ("c0", "c1", "c2") else \
            ("y" if rng.random() < 0.9 else "x")
        rows.append({"cat": v, "tgt": tgt, "http.route": "r"})
    pr = _probe(rows)
    hier = _hier({"cat": "categorical", "tgt": "categorical", "http.route": "categorical"})
    vg = SEL.value_groups(pr, T0 + 3000, hier, "cat", ["tgt"])
    assert len(set(vg.values())) == 2
    assert vg["c0"] == vg["c1"] == vg["c2"] != vg["c3"] == vg["c4"] == vg["c5"]
