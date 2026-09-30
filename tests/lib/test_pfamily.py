"""lib/pfamily: system families (progressive.md §6.20)."""
from __future__ import annotations

import time

import numpy as np
import pytest

from app.engines.behavior.lib import pfamily as PF


def groups_for(app: str, host: str, n_routes: int = 6, stack: str = "chrome", extra_routes=()):
    routes = {f"{host} /{app}/r{i}/*": float(10 + i) for i in range(n_routes)}
    routes.update({r: 5.0 for r in extra_routes})
    return {"route": routes, "host": {host: 1.0}, "port": {"8080": 1.0}, "stack": {stack: 1.0},
            "name": {n: 1.0 for n in ("net.src", "net.dst", "http.route", "http.method", "body.len")}}


def sig_of(groups, pv=0.3):
    fs = PF.feature_set(groups)
    return {"fs": fs, "sig": PF.weighted_minhash(fs), "informative": PF.informative(groups),
            "payload_vis": pv}


def test_route_prefix_and_etld():
    assert PF.route_prefix("POST oa /approval/{num}/approve") == "oa /approval/{num}/*"
    assert PF.route_prefix("GET oa /login") == "oa /login"
    assert PF.etld1("mail.corp.example.com") == "example.com"


def test_weighted_minhash_estimates_weighted_jaccard():
    r = np.random.default_rng(0)
    a = {f"f{i}": float(r.uniform(0.1, 1.0)) for i in range(60)}
    b = dict(a)
    for i in range(0, 60, 3):
        b[f"f{i}"] *= 0.3
    for i in range(60, 80):
        b[f"f{i}"] = 0.5
    j = PF.weighted_jaccard(a, b)
    est = PF.minhash_similarity(PF.weighted_minhash(a, k=512), PF.weighted_minhash(b, k=512))
    assert abs(est - j) < 0.07, (j, est)


def test_generic_features_do_not_make_systems_similar():
    oa = PF.feature_set(groups_for("oa", "oa.corp"))
    fin = PF.feature_set(groups_for("fin", "fin.corp"))
    rep = PF.feature_set(groups_for("oa", "oa.corp"))
    assert PF.weighted_jaccard(oa, fin) < 0.4          # same names / port / stack, other app
    assert PF.weighted_jaccard(oa, rep) > 0.95


def test_family_forms_after_two_days_and_is_stable():
    st = {}
    sigs = {"oa": sig_of(groups_for("oa", "oa.corp")), "oa-r2": sig_of(groups_for("oa", "oa.corp")),
            "oa-r3": sig_of(groups_for("oa", "oa.corp", extra_routes=["oa.corp /x/*"])),
            "fin": sig_of(groups_for("fin", "fin.corp"))}
    r1 = PF.update_families(st, 100, sigs)
    assert r1["member"] == {}                          # one day is not enough
    r2 = PF.update_families(st, 101, sigs)
    fam = r2["member"]
    assert set(fam) == {"oa", "oa-r2", "oa-r3"} and len(set(fam.values())) == 1
    assert "fin" not in fam
    fid = fam["oa"]
    # idle members keep their family; ids are stable across runs
    r3 = PF.update_families(st, 102, {"oa": sigs["oa"], "fin": sigs["fin"]})
    assert r3["member"] == fam and r3["joined"] == []
    # a new replica joins on its second daily match, into the same family id
    sigs["oa-r4"] = sig_of(groups_for("oa", "oa.corp"))
    PF.update_families(st, 103, sigs)
    r5 = PF.update_families(st, 104, sigs)
    assert r5["member"]["oa-r4"] == fid and ("oa-r4", fid) in r5["joined"]


def test_non_consecutive_days_do_not_count_and_payload_mismatch_blocks():
    st = {}
    a, b = sig_of(groups_for("x", "x.corp"), 0.3), sig_of(groups_for("x", "x.corp"), 0.9)
    PF.update_families(st, 1, {"a": a, "b": b})
    assert PF.update_families(st, 2, {"a": a, "b": b})["member"] == {}     # payload differs by 0.6
    st = {}
    b = sig_of(groups_for("x", "x.corp"), 0.3)
    PF.update_families(st, 1, {"a": a, "b": b})
    assert PF.update_families(st, 3, {"a": a, "b": b})["member"] == {}     # day 2 missing


def test_uninformative_signatures_never_match():
    st = {}
    idle = {f"idle{i}": sig_of({"route": {f"idle{i}.corp /status": 1.0}, "port": {"8080": 1.0},
                                "name": {"net.src": 1.0}}) for i in range(5)}
    for d in range(5):
        r = PF.update_families(st, d, idle)
    assert r["member"] == {}


def test_config_force_and_forbid():
    st = {}
    sigs = {"a": sig_of(groups_for("x", "x.corp")), "b": sig_of(groups_for("x", "x.corp")),
            "c": sig_of(groups_for("y", "y.corp"))}
    cfg = [{"name": "pair", "systems": ["a", "c"], "force": True}, {"systems": ["b"], "force": False}]
    r = PF.update_families(st, 1, sigs, cfg)
    assert r["member"] == {"a": "fam:pair", "c": "fam:pair"}
    r = PF.update_families(st, 2, sigs, cfg)
    assert "b" not in r["member"]


def test_detach_after_seven_days_over_share():
    st = {}
    sigs = {s: sig_of(groups_for("x", "x.corp")) for s in ("a", "b", "c")}
    PF.update_families(st, 1, sigs)
    r = PF.update_families(st, 2, sigs)
    fid = r["member"]["a"]
    left = []
    for d in range(3, 12):
        r = PF.update_families(st, d, sigs, detach_shares={"a": 0.3, "b": 0.05, "c": 0.0})
        left += [x for x in r["left"] if x[0] == "a"]
        if left:
            break
    assert left and left[0][1] == fid and d == 2 + PF.DETACH_DAYS
    assert "a" not in r["member"] and r["member"]["b"] == fid
    r = PF.update_families(st, d + 1, sigs)                 # no immediate rejoin
    assert "a" not in r["member"]


def test_family_pass_cost_is_linear_in_systems():
    """LSH candidates, not all pairs: 12 families x 25 members + 30 singletons."""
    def sigs_n(n_fam, per):
        out = {}
        for f in range(n_fam):
            for m in range(per):
                out[f"app{f}-{m}"] = sig_of(groups_for(f"app{f}", f"app{f}.corp"))
        for i in range(30):
            out[f"single{i}"] = sig_of(groups_for(f"s{i}", f"s{i}.corp"))
        return out
    t = {}
    for per in (2, 8, 25):
        sigs = sigs_n(12, per)
        st = {}
        PF.update_families(st, 1, sigs)
        t0 = time.perf_counter()
        r = PF.update_families(st, 2, sigs)
        t[per] = (time.perf_counter() - t0) / len(sigs)
        assert len(r["families"]) == 12 and all(len(m) == per for m in r["families"].values())
    # per-system cost of the pass stays flat within a factor 3 (pairs only inside families)
    assert t[25] <= 3.0 * t[2] + 2e-3, t
