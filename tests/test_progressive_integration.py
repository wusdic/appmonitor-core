"""Integration of the progressive profile core (docs/lib3/progressive.md §9)
and regression tests for the root causes the integrator found on pack O.

Each test names the measured failure it guards against; the numbers come from
the integration runs recorded in docs/lib3/progressive.md §16."""
from __future__ import annotations

import json
import math
import pickle

import numpy as np
import pytest

from app.core.engine import DEFAULT_CONFIG, default_config
from app.core.store import MetricStore
from app.engines.behavior import pattern_tree as P4
from app.engines.behavior import risk as B26
from app.engines.behavior.lib import detectors as DET
from app.engines.behavior.lib import pbounds as PB
from app.engines.behavior.lib import pdfg as DF
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import phier as PH
from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import pparse as PP
from app.engines.behavior.lib import psketch as PS
from app.engines.behavior.lib import stages as STG
from app.eval import pmetrics as PM
from app.eval.runner import _jsonable
from app.pipeline import build


# ------------------------------------------------------------ registry (§9.1)
def test_registry_modes_follow_section_9_1_order():
    full = [e.name for e in build.build_registry(progressive="full").ordered()]
    assert not any(n in full for n in ("raw.event", "derived.event_context", "behavior.pattern_tree"))
    prog = [e.name for e in build.build_registry(progressive="full+progressive").ordered()]
    # the full registry is a subsequence of the progressive one (nothing reordered)
    it = iter(prog)
    assert all(n in it for n in full)
    idx = {n: i for i, n in enumerate(prog)}
    order = ["raw.client_stack", "raw.event", "derived.session", "derived.event_context",
             "behavior.resource_governor", "behavior.feature_vector", "behavior.cross_system",
             "behavior.attr_registry", "behavior.attr_select", "behavior.conformity",
             "behavior.pattern_tree", "behavior.content_bounds", "behavior.payload_grammar",
             "behavior.binding", "behavior.time_window", "behavior.workflow",
             "behavior.who_groups", "behavior.system_profile", "behavior.feedback",
             "behavior.calibration", "behavior.fusion", "behavior.risk", "behavior.incident",
             "behavior.governor", "behavior.explain", "behavior.facets", "behavior.views",
             "behavior.portrait", "signature.rule_match"]
    assert [idx[n] for n in order] == sorted(idx[n] for n in order)
    only = [e.name for e in build.build_registry(progressive="progressive_only").ordered()]
    assert only[:4] == ["raw.action_token", "raw.client_stack", "raw.event", "derived.event_context"]
    assert not any(n.startswith("signature.") or n == "behavior.feature_vector" for n in only)


def test_registry_mode_resolution():
    class _Pack:
        registry_mode = "progressive_only"
    assert build.registry_mode() == "full"
    assert build.registry_mode(config={"progressive": {"enabled": True}}) == "full+progressive"
    assert build.registry_mode(pack=_Pack()) == "progressive_only"
    assert build.registry_mode(False, {"progressive": {"enabled": True}}) == "full"
    with pytest.raises(ValueError):
        build.registry_mode("everything")


def test_default_config_keys_of_section_13_2():
    cfg = default_config()
    assert cfg["progressive"] == {"enabled": False}
    assert cfg["who_group_names"] == [] and cfg["lib3"]["resource_mode"] == "full"
    # readers deep-merge over the full defaults
    assert EV.pconfig(cfg)["defaults"]["body_cap"] == 4096
    assert "progressive" in DEFAULT_CONFIG


# ---------------------------------------------------- detectors, B25-B29 (§9.2)
def test_conformity_detectors_and_pattern_violation_kind_are_wired():
    for d in ("conf_who", "conf_when", "conf_content", "conf_seq", "conf_novel"):
        assert DET.DETECTOR_INFO[d]["family"] == "conformity" and DET.is_instant(d)
    assert DET.N_V2_DETECTORS == 31 and DET.DETECTORS[30] == "cross_system"
    assert STG.stages_for(["content"]) == {"behavior"}
    from app.engines.behavior import explain, fusion, incident
    for mod in (incident, fusion, explain):
        assert "pattern_violation" in mod.DISCRETE_KINDS
    assert "pattern_violation" in B26.EVENT_KINDS


@pytest.mark.parametrize("typ,sev,flags,w", [
    ("who", "high", [], 15.0), ("who", "medium", [], 8.0), ("who", "low", [], 3.0),
    ("content", "medium", ["cross_binding"], 10.0), ("content", "medium", ["injection_shape"], 8.0),
    ("novel", "medium", [], 8.0), ("seq", "info", [], 0.0)])
def test_risk_stage_weights_of_pattern_violations(typ, sev, flags, w):
    """progressive.md §9.3 B26 row: who HIGH-candidate 15, MEDIUM 8, LOW 3;
    binding MEDIUM 10; content MEDIUM 8; novel MEDIUM 8."""
    assert B26.pv_weight({"type": typ, "flags": flags}, sev) == w


# --------------------------------------------------------- snapshots, pickling
def test_runner_snapshots_render_tree_nodes_as_plain_data():
    """pnode.Node has __slots__: the runner used to render it as a repr string,
    so pmetrics.false_splits saw no node contexts (open issue of W-P3)."""
    nd = PN.Node(3, 0, 1, 0, (("net.src", 1, frozenset({"192.168.1.0/24"}), False),), 0.0)
    out = _jsonable({"nodes": {3: nd}})
    assert out["nodes"]["3"]["ctx"] == [["net.src", 1, ["192.168.1.0/24"], False]]
    assert json.dumps(out)
    tree = {"kinds": {"0": {"nodes": out["nodes"]}}}
    assert PM._ptree_contexts(tree) == [("3", [["net.src", 1, ["192.168.1.0/24"], False]])]


def test_metric_store_pickles_without_its_lock():
    st = MetricStore()
    st.put_model("oa", "__system__", "model.x", {"a": 1}, ts=0.0)
    st2 = pickle.loads(pickle.dumps(st))
    assert st2.get_model("oa", "__system__", "model.x") == {"a": 1}
    st2.put_model("oa", "__system__", "model.y", 2, ts=1.0)        # lock recreated


# ------------------------------------------------------------ foundation fixes
def test_route_key_reads_absent_attributes_as_missing():
    """ABSENT is the string '⊥'; route_key returned 'DST ⊥' / 'TLS ⊥' for rows
    without the attribute (P03 / P11 / P14 each carried a local wrapper)."""
    assert DF.route_key(lambda a: EV.ABSENT) is None
    got = DF.route_key(lambda a: {"net.dst": "10.0.0.1:443"}.get(a, EV.ABSENT))
    assert got == "DST 10.0.0.1:443"


def test_json_body_cut_at_the_capture_cap_keeps_its_leading_keys():
    """综合部's 20-60 KB reports are cut at BODY_CAP = 4096 bytes: the strict
    parser produced no keys at all, so body.keys of GA.oa.report was empty."""
    body = json.dumps({"dept": "GA", "period": "2025-09", "items": ["x" * 40] * 400, "remark": "r"})
    out = PP.parse_body(body[:PP.BODY_CAP], "application/json")
    assert {"dept", "period"} <= set(out["body.keys"])
    assert "remark" not in out["body.keys"]                    # after the cut: never seen
    assert PP.json_prefix('{"a": "x,y", "b": [1, 2, 3') == {"a": "x,y", "b": [1, 2]}
    with pytest.raises(ValueError):
        PP.json_prefix("{not json")


def test_damped_rows_do_not_move_the_hard_range():
    """A3's single 12 KB login, learned with outlier damping, widened a node's
    stated maximum to 12 KB (W-P4 open issue): damped rows feed the digest
    but not the daily extreme ring or the exceedance reservoirs."""
    s = PN.NumSummary()
    for i in range(200):
        s.update(1000.0 + i, float(i), 1.0, 1.0, day=10)
    before = s.observed_range(10)[:2]
    s.update(12000.0, 300.0, 0.2, 0.2, day=10, extreme=False)
    assert s.observed_range(10)[:2] == before
    s.update(12000.0, 301.0, 1.0, 1.0, day=10)                   # an undamped row does
    assert s.observed_range(10)[1] == 12000.0


def test_constant_attribute_band_has_full_coverage():
    """A constant attribute (net.pkts_down = 2 on every login) has band [2, 2];
    open-interval coverage was 0, statement confidence 1e-308 and every OA
    statement read "置信 0.00"."""
    td = PS.TDigest(50.0, PS.H_M)
    for _ in range(100):
        td.add(2.0, 0.0, 1.0)
    assert PB.closed_coverage(td, 2.0, 2.0) == pytest.approx(1.0)
    rec = PB.fit_digest(td, 0.0, n=100.0)
    assert rec["confidence"] > 0.9


# ------------------------------------------------------------------ P04 fixes
def test_coder_hash_bins_a_flat_high_cardinality_target():
    """29 usernames with no dominant value: the top-8 bins held < 30 % of the
    logins, so the username target could not tell 综合部 from 销售部."""
    seeds = [[(f"user{i:02d}", 1.0 / 29) for i in range(16)], [("form", 0.99)]]
    c = P4.Coder(["body.kv.username", "body.fmt"], seeds)
    assert c.hashed == [True, False]
    b1 = c.bins(["jack", "form"])
    assert b1 == c.bins(["jack", "form"]) and 0 <= b1[0] < P4.K_B
    assert c.value_of(0, b1[0]) is None and c.value_of(1, b1[1]) == "form"
    assert P4._stable_bucket("jack", 9) == P4._stable_bucket("jack", 9)


class _LCStub:
    def __init__(self) -> None:
        self.hier = PH.Hierarchies()
        self.gone = set()
        self.now = 0.0


def _cands(ctx, sc):
    nd = PN.Node(1, 0, len(ctx), 0, tuple(ctx), 0.0)
    eng = P4.PatternTreeEngine()
    return eng._leaf_cands(_LCStub(), None, nd, 0, {"split_cands": {0: sc}})


def test_a_value_group_can_be_divided_at_its_own_level():
    """The OA root put 192.168.1/2/3.0/24 into ONE group; /24 was then never
    offered below it, so 综合部's, 财务部's and 销售部's logins stayed one node."""
    grp = ("net.src", 1, frozenset({"192.168.1.0/24", "192.168.2.0/24"}), False)
    one = ("net.src", 1, frozenset({"192.168.1.0/24"}), False)
    sc = [("net.src", 1), ("net.src", 3)]
    assert ("net.src", 1) in _cands([grp], sc)
    assert ("net.src", 1) not in _cands([grp, one], sc)          # a single value: done
    route_grp = ("http.route", 2, frozenset({"h /login", "h /home"}), False)
    assert ("http.route", 2) in _cands([route_grp], [("http.route", 2)])


def test_route_facet_keeps_a_candidate_slot():
    """Ranked by U_s the route family came third behind a size bin and a
    user-agent shape, so OA never split on the action."""
    sc = [("net.src", 1), ("net.src", 3), ("ctx.tod_min", 2), ("ctx.tod_min", 3),
          ("net.bytes_up", 1), ("hdr.user-agent", 3), ("http.route", 2), ("net.pkts_down", 1)]
    got = _cands([], sc)
    assert ("http.route", 2) in got and len(got) == P4.C_MAX


# ------------------------------------------------------------------ P05 fix
def test_node_targets_admit_node_local_attributes():
    """The login form's username covers < 5 % of OA's events, so it never
    entered the system target list and no node could target it."""
    from app.engines.behavior.lib import pselect as SEL
    from app.engines.behavior.lib import ptree as PT
    probe = SEL.StratifiedProbe(512)
    rng = np.random.default_rng(0)
    for i in range(300):
        login = i % 10 == 0
        row = {"http.route": "POST h /login" if login else "GET h /docs",
               "net.bytes_up": float(rng.integers(400, 2000))}
        if login:
            row["body.kv.username"] = f"u{int(rng.integers(0, 12))}"
        probe.offer("all", row, 1.0, float(i), float(rng.random()))
    tr = PT.Tree(0, 0.0)
    tr.split(tr.root, "http.route", 0, [["POST h /login"]], 0.0)
    login_nid = tr.nodes[tr.root].split.children[0]
    hier = PH.Hierarchies()
    no_pool = SEL.node_targets_from_probe(tr, probe, 300.0, hier, ["net.bytes_up"], n_min=8)
    pool = SEL.node_targets_from_probe(tr, probe, 300.0, hier, ["net.bytes_up"], n_min=8,
                                       local_pool=["body.kv.username"])
    assert "body.kv.username" not in (no_pool.get(login_nid) or [])
    assert "body.kv.username" in pool[login_nid]
    assert "body.kv.username" not in (pool.get(tr.root) or [])       # 10 % coverage there


# ------------------------------------------------------------------ scorer fixes
def test_scorer_normalises_array_keys_and_numeric_strings():
    assert PM._keyset(["lines[]", "dept"]) == {"lines", "dept"}
    assert PM._num("004217") == 4217.0 and PM._num("abc") is None


def test_scorer_draws_held_out_minutes_from_the_generators_arrival_law():
    row = {"tid": "x", "who": {"level": "any"}, "windows": {"workday": [[420, 1380]]},
           "gen": {"arrival": "normal", "members": {"1.2.3.4": 1.0}}}
    r = np.random.default_rng(0)
    evs = PM.holdout_events([row], r, 400)
    ms = np.asarray([e["minute"] for e in evs])
    # N(900, 240) clipped: about 68 % within one sd, not the uniform 50 %
    assert 0.6 < np.mean(np.abs(ms - 900) <= 240) < 0.76
