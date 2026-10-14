"""lib/pfd, round 7 (time / workflow / content / views owner): P08's memory
grows with the binding EVIDENCE, not with the attribute count (PG4 memory
slope vs attributes). Fail on round 6 (52e813f)."""
from __future__ import annotations

import numpy as np

from app.engines.behavior.lib import pfd as FD

DEPTS = {"10.0.1.%d" % i: "ga" for i in range(1, 6)}
DEPTS.update({"10.0.2.%d" % i: "fin" for i in range(1, 6)})


def _rows(n_copies: int):
    """Rows of ten addresses whose department-valued attributes (n_copies of
    them, each a function of the department: the same partition of the rows)
    and user name (a function of the address)."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(300):
        ip = sorted(DEPTS)[int(rng.integers(0, len(DEPTS)))]
        r = {"net.src": ip, "body.kv.username": "u" + ip.split(".")[-1] + ip.split(".")[-2]}
        for j in range(n_copies):
            r[f"meta.f{j:03d}"] = f"{DEPTS[ip]}-{j % 7}"
        rows.append(r)
    return rows


def _screen(rows, ys, card):
    gen = (lambda a, l, v: ".".join(str(v).split(".")[:3]) + ".0/24" if l == 1 else v)
    return FD.screen(rows, np.ones(len(rows)), [("net.src", 0), ("net.src", 1)], ys, gen,
                     card=lambda a: card.get(a, 0.0))


def test_payloads_with_the_same_partition_are_screened_once():
    rows = _rows(6)
    ys = ["body.kv.username"] + [f"meta.f{j:03d}" for j in range(6)]
    card = {"body.kv.username": 10.0, **{f"meta.f{j:03d}": 2.0 for j in range(6)}}
    specs = _screen(rows, ys, card)
    payloads = {(d["x"], d["y"] if d["dir"] == "fwd" else d["x"]) for d in specs}
    fwd = {(d["x"], d["y"]) for d in specs if d["dir"] == "fwd"}
    assert ("net.src", "body.kv.username") in fwd                 # the identifier keeps its pair
    dept = {d["y"] for d in specs if d["dir"] == "fwd" and d["y"].startswith("meta.")}
    assert dept == {"meta.f000"}                                  # one department attribute, not six
    assert payloads


def test_a_pair_record_keeps_only_sources_with_evidence():
    table = {
        "10.0.0.1": {"n": 6.0, "top": "jack", "k": 6.0, "mass": 6.0, "LB": 0.9, "a0": 1.0, "b0": 0.1,
                     "p_viol": 0.02, "bound": True},
        "10.0.0.2": {"n": 3.0, "top": "rose", "k": 3.0, "mass": 3.0, "LB": 0.5, "a0": 1.0, "b0": 0.1,
                     "p_viol": 0.1},                              # recurring: a binding in the making
        "10.0.0.3": {"n": 1.0, "top": "x", "k": 1.0, "mass": 1.0, "LB": 0.2, "a0": 1.0, "b0": 0.1,
                     "p_viol": 0.4},                              # a one-off visitor
        "10.0.0.4": {"n": 1.0, "top": "y", "k": 1.0, "mass": 1.0, "LB": 0.2, "a0": 1.0, "b0": 0.1,
                     "p_viol": 0.4},                              # bound at the previous fit (rename)
        "10.0.0.5": {"n": 1.0, "top": "z", "k": 1.0, "mass": 1.0, "set": ["z", "w"], "U": 0.01},
    }
    out = FD.compact_table(table, keep=["10.0.0.4"])
    assert set(out) == {"10.0.0.1", "10.0.0.2", "10.0.0.4", "10.0.0.5"}
    assert all(not ({"k", "mass", "a0", "b0"} & set(e)) for e in out.values())
    rec = {"fd": {"holds": True}, "table": out, "bound_values": {"jack": ["10.0.0.1"]}}
    assert FD.check_forward(rec, "10.0.0.1", "rose") == (FD.check_forward(
        {"fd": {"holds": True}, "table": table, "bound_values": {"jack": ["10.0.0.1"]}}, "10.0.0.1", "rose"))
    assert FD.check_forward(rec, "10.0.0.3", "x")[1] == []        # unknown source: no check, as unbound
