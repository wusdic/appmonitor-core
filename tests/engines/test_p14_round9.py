"""P14 round 9: one statement per configured department stated in pieces
(OI8-a / OI8-g).

Pack O seeds 1-4, round 8, day 14: GA.mail#0 missed on who - 综合部's mail was
stated by the 192.168/16 node (.21, .23) and the 10.168/16 node (.121), never
as 综合部; FIN.mail#0 - 财务部's parts split by time of day. A department
statement composes such pieces: who = union, when = union of the windows,
content = what holds for every piece, confidence = the minimum. Pieces of
different departments, or statements mixing departments, are never merged,
and nothing is stated when one statement already lists the members. Fails on
round 8 (no department_statements)."""
from __future__ import annotations

from app.engines.behavior import views as VW
from app.eval import pmetrics as PMX

GA = {"192.168.1.21": "综合部", "192.168.1.23": "综合部", "10.168.7.121": "综合部",
      "192.168.2.10": "财务部", "192.168.2.11": "财务部"}


def dept_of(ips):
    ds = {GA.get(i) for i in ips}
    return next(iter(ds)) if len(ds) == 1 and None not in ds else None


def _st(nid, members, wd, band, conf, ctx_extra=(), state="stable", act=4, closed=None, part=None):
    content = {"net.bytes_down": {"band90": list(band), "band90_raw": list(band), "coverage": 0.9,
                                  "range": [band[0] * 0.8, band[1] * 1.2], "cover": 0.1, "hard": True,
                                  "n_rng": 20}}
    if closed is not None:
        content["tls.ver"] = {"closed": list(closed), "U": 0.02}
    ctx = [["tls.sni", 0, ["mail.corp.local"], False]] + [list(x) for x in ctx_extra]
    pid = f"p:mail:0:{nid}@1.0" + (f"|grp:{part}" if part else "")
    return {"id": pid, "pattern_id": pid, "state": state, "confidence": conf, "support": 30.0,
            "n_obs": 30.0, "mass": 10.0, "first_seen": 1.7e9, "last_seen": 1.7e9 + 86400 * 10,
            "version": 1, "cver": 0, "act_node": act,
            "evidence": {"route": "TLS mail.corp.local", "system": "mail", "kind": 0, "node": nid,
                         "depth": 2, "context": ctx,
                         "who": {"level": "grp" if part else "ip", "items": list(members),
                                 "members": list(members), "closed": not part},
                         "when": {"workday": [list(w) for w in wd], "nonworkday": [], "coverage": 0.85,
                                  "confidence": 0.8},
                         "content": content, "bindings": {}, "workflow": [],
                         "facet_conf": {"who": conf, "when": conf}}}


SIB_A = ["net.src", 1, ["192.168.0.0/16"], False]
SIB_B = ["net.src", 1, ["10.168.0.0/16"], False]


def test_one_department_across_sibling_source_nodes():
    a = _st(7, ["192.168.1.21", "192.168.1.23"], [[540, 600]], (2000.0, 60000.0), 0.9, [SIB_A],
            closed=["1.2"])
    b = _st(8, ["10.168.7.121"], [[555, 620], [800, 830]], (3000.0, 70000.0), 0.8, [SIB_B],
            closed=["1.3"])
    fin = _st(9, ["192.168.2.10", "192.168.2.11"], [[600, 660]], (1.0, 2.0), 0.95, [SIB_A])
    out = VW.department_statements([a, b, fin], dept_of, "mail")
    assert len(out) == 1
    d = out[0]
    ev = d["evidence"]
    assert ev["who"]["members"] == ["10.168.7.121", "192.168.1.21", "192.168.1.23"]
    assert ev["who"]["dept"] == "综合部" and ev["who"]["closed"] is False
    assert ev["when"]["workday"] == [[540.0, 620.0], [800.0, 830.0]]
    assert ev["content"]["net.bytes_down"]["band90"] == [2000.0, 70000.0]
    assert ev["content"]["tls.ver"]["closed"] == ["1.2", "1.3"]
    assert d["confidence"] == 0.8 and d["state"] == "stable"
    # judged on the members' own events, in the action's shared context
    assert ev["context"] == [["tls.sni", 0, ["mail.corp.local"], False],
                             ["net.src", 0, ["10.168.7.121", "192.168.1.21", "192.168.1.23"], False]]
    assert "综合部（10.168.7.121、192.168.1.21、192.168.1.23）" in d["text_zh"]
    # the scorer reads it as 综合部's who (GA.mail#0: grp truth, Jaccard >= 0.8)
    ls = PMX.LStmt(d, "mail", {})
    assert PMX.who_compatible({"level": "grp", "members": ["192.168.1.21", "192.168.1.23", "10.168.7.121"]},
                              ls.who)
    assert ls.confirmed


def test_parts_of_one_node_split_by_time_and_by_learned_sub_groups():
    # 财务部's mail: .11 in the afternoon sibling, .10 in the morning sibling
    am = _st(11, ["192.168.2.10"], [[540, 600]], (1.0, 2.0), 0.7, [["ctx.tod_min", 3, ["am"], False]],
             part="G3")
    pm = _st(12, ["192.168.2.11"], [[840, 900]], (1.0, 3.0), 0.9, [["ctx.tod_min", 3, ["pm"], False]],
             part="G4")
    out = VW.department_statements([am, pm], dept_of, "mail")
    assert len(out) == 1
    ev = out[0]["evidence"]
    assert ev["who"]["members"] == ["192.168.2.10", "192.168.2.11"]
    assert ev["when"]["workday"] == [[540.0, 600.0], [840.0, 900.0]]
    assert out[0]["confidence"] == 0.7


def test_no_merge_across_departments_contexts_or_when_already_stated():
    a = _st(7, ["192.168.1.21", "192.168.1.23"], [[540, 600]], (1.0, 2.0), 0.9, [SIB_A])
    b = _st(8, ["10.168.7.121"], [[555, 620]], (1.0, 2.0), 0.8, [SIB_B])
    mixed = _st(9, ["10.168.7.121", "192.168.2.10"], [[555, 620]], (1.0, 2.0), 0.8, [SIB_B])
    # a statement mixing departments is no piece
    assert VW.department_statements([a, mixed], dept_of, "mail") == []
    # a piece under a content condition is a variant: another context
    c = _st(8, ["10.168.7.121"], [[555, 620]], (1.0, 2.0), 0.8, [["http.len", 2, ["b3"], False]])
    assert VW.department_statements([a, c], dept_of, "mail") == []
    # another action
    d = dict(b, act_node=5)
    assert VW.department_statements([a, d], dept_of, "mail") == []
    # the action node already states all three: no duplicate
    full = _st(4, ["192.168.1.21", "192.168.1.23", "10.168.7.121"], [[540, 620]], (1.0, 2.0), 0.9,
               part="G1")
    assert VW.department_statements([a, b, full], dept_of, "mail") == []
    # a stale or a time-facet statement is no piece
    assert VW.department_statements([a, dict(b, state="stale")], dept_of, "mail") == []
    wf = dict(b, pattern_id=b["pattern_id"] + "|~when")
    assert VW.department_statements([a, wf], dept_of, "mail") == []


def test_content_that_holds_for_every_piece():
    m = VW.merge_content([
        {"a": {"band90": [1, 2], "coverage": 0.9}, "k": {"required": ["x", "y"]}, "g": {"grammar": "[a-z]{4}"},
         "only": {"closed": ["1"]}},
        {"a": {"band90": [1.5, 3], "coverage": 0.8}, "k": {"required": ["x"]}, "g": {"grammar": "[a-z]{5}"}}])
    assert m["a"]["band90"] == [1.0, 3.0] and m["a"]["coverage"] == 0.8
    assert m["k"]["required"] == ["x"]
    assert m["g"]["grammar"] == "(?:[a-z]{4}|[a-z]{5})"
    assert "only" not in m
    assert VW.merge_when([{"workday": [[1, 5]], "coverage": 0.9}, None]) is None


# ------------------------------------------------------------------ OI8-j
class _N:
    def __init__(self, prior=None, iss=None, hold=None):
        self.meta = {}
        if prior is not None:
            self.meta["hold_prior"] = prior
        if iss is not None:
            self.meta["hold_prior_iss"] = iss
        self.hold = hold

    def p_hold(self, t=None):
        pr = self.meta.get("hold_prior")
        return None if pr is None else round(pr[0] / (pr[0] + pr[1]), 4)


def test_a_statement_confirmed_before_p04s_daily_step_states_the_pooled_rate():
    """OI8-j, pack O seed 0 day 7: oa approval nodes confirmed during the day
    had neither a held-out test nor a prior (P04 issues it at its daily step),
    and their views (22:15 / 23:45) stated the minimum of the clauses'
    nominal values - 0.0547, rate.ip_h's coverage lower bound from ~1
    IP-hour - and held. The tree's pooled rate is stated instead."""
    from types import SimpleNamespace as NS
    fresh = _N()
    tree = NS(nodes={1: _N((3.5, 0.5), (0, "*")), 2: _N((9.0, 1.0), (0, 3)), 3: fresh,
                     4: _N((1.0, 1.0), None, hold=object())})
    c = NS(ptm=NS(kinds={0: tree}))
    b = VW.base_hold(c)
    assert abs(b - 0.875) < 1e-12                  # the '*' issuance only
    assert VW.compose_conf(fresh, 0.0, 0.0547, None, False, None, None, b) == 0.875
    # a group's part with its own windows: min(pooled rate, the windows' hold)
    part = ("G10", ["10.0.0.1"], "G10", 0.3)
    assert VW.compose_conf(fresh, 0.0, 0.0, part, True, 0.0, 0.69, b) == 0.69
    # without any issued prior the nominal fallback stays
    c2 = NS(ptm=NS(kinds={0: NS(nodes={3: _N()})}))
    assert VW.base_hold(c2) is None
    assert VW.compose_conf(fresh, 0.0, 0.0547, None, False, None, None, VW.base_hold(c2)) == 0.0547
    # untested nodes with a kind prior only: their mean
    c3 = NS(ptm=NS(kinds={0: NS(nodes={2: _N((9.0, 1.0), (0, 3)), 5: _N((3.0, 1.0))})}))
    assert abs(VW.base_hold(c3) - (0.9 + 0.75) / 2) < 1e-12


# ------------------------------------------------------------------ OI8-g
class _Lvl:
    def __init__(self, items):
        self._items = items

    def total(self, t):
        return float(sum(c for _, c in self._items))

    def items(self, t, *a):
        return [(k, c, c, 0.0) for k, c in self._items]


def _shared_node(ga_extra_group=True):
    from types import SimpleNamespace as NS
    # O-real fam:1, day 35: P11's role group G18 holds 综合部's .21 with 财务部's
    # clerks .2.12 / .2.51; G19 holds 综合部's .121 / .23
    groups = {"G18": {"name": "G18·fam:1 GET /flow/list", "members": ["192.168.1.21", "192.168.2.12",
                                                                      "192.168.2.51"]},
              "G19": {"name": "综合部", "dept": "综合部", "members": ["10.168.7.121", "192.168.1.23"]},
              "G20": {"name": "财务部", "dept": "财务部", "members": ["192.168.2.10", "192.168.2.11"]}}
    ip2g = {m: g for g, gr in groups.items() for m in gr["members"]}
    cfg = {"who_group_names": [
        {"name": "综合部", "ips": ["192.168.1.21", "192.168.1.23", "10.168.7.121"]},
        {"name": "财务部", "ips": ["192.168.2.10", "192.168.2.11", "192.168.2.12", "192.168.2.51"]}]}
    c = NS(ip2g=ip2g, mode="grp", groups=groups, sigs=None, key="fam:1", config=cfg)
    ips = [(m, 4.0) for gr in groups.values() for m in gr["members"]]
    nd = NS(ctx=[], who=NS(levels=[_Lvl(ips), _Lvl([]), _Lvl([]),
                                   _Lvl([("grp:G18", 12.0), ("grp:G19", 8.0), ("grp:G20", 8.0)])]))
    return c, nd


def test_configured_departments_are_parts_of_a_node_shared_with_mixed_learned_groups():
    """OI8-g, O-real seed 0 day 35: ~20 of 30 misses failed on who only - the
    shared fam:1 route nodes were stated per learned role group, and G18
    mixes 综合部's .21 with 财务部's clerks, so no part was 综合部 or 财务部.
    Each configured department is now a part of its own (its listed
    addresses seen at the node); the learned parts stay."""
    c, nd = _shared_node()
    parts = VW.group_parts(c, nd, 0.0, 0.0, "GET oa /docs")
    by = {p[0]: p for p in parts}
    assert by["cfg:综合部"][1] == ["10.168.7.121", "192.168.1.21", "192.168.1.23"]
    assert by["cfg:财务部"][1] == ["192.168.2.10", "192.168.2.11", "192.168.2.12", "192.168.2.51"]
    assert by["cfg:综合部"][4] == ()
    assert {"G18", "G19", "G20"} <= set(by)


def test_no_configured_part_when_the_learned_groups_already_state_the_department():
    from types import SimpleNamespace as NS
    groups = {"G6": {"name": "综合部", "dept": "综合部", "members": ["10.168.7.121", "192.168.1.23"]},
              "G25": {"name": "综合部·x", "dept": "综合部", "members": ["192.168.1.21"]},
              "G10": {"name": "销售部", "dept": "销售部", "members": ["192.168.3.20"]}}
    ip2g = {m: g for g, gr in groups.items() for m in gr["members"]}
    c = NS(ip2g=ip2g, mode="grp", groups=groups, sigs=None, key="oa",
           config={"who_group_names": [{"name": "综合部", "ips": ["192.168.1.21", "192.168.1.23",
                                                                  "10.168.7.121"]}]})
    ips = [(m, 4.0) for gr in groups.values() for m in gr["members"]]
    nd = NS(ctx=[], who=NS(levels=[_Lvl(ips), _Lvl([]), _Lvl([]),
                                   _Lvl([("grp:G6", 8.0), ("grp:G25", 4.0), ("grp:G10", 4.0)])]))
    parts = VW.group_parts(c, nd, 0.0, 0.0, "GET oa /docs")
    assert not any(p[0].startswith("cfg:") for p in parts)
    assert next(p for p in parts if p[0] == "dept:综合部")[1] == ["10.168.7.121", "192.168.1.21", "192.168.1.23"]


def test_department_windows_refitted_on_the_members_arrivals():
    """With the lowest common node's arrivals (when_of), the department states
    the windows fitted on all its members' arrivals, at their hold; pack O
    seed 1, days 15-21: the .121 piece's own windows were a 0.5-coverage
    day span, so the union of the pieces' windows matched no truth window."""
    a = _st(7, ["192.168.1.21", "192.168.1.23"], [[810, 845]], (2000.0, 60000.0), 0.9, [SIB_A])
    b = _st(8, ["10.168.7.121"], [[0, 942], [1410, 1440]], (3000.0, 70000.0), 0.95, [SIB_B])
    seen = []

    def when_of(pieces, members):
        seen.append((sorted(p["pattern_id"] for p in pieces), list(members)))
        return ({"workday": [[540, 570], [810, 845]], "nonworkday": [], "coverage": 0.9, "confidence": 0.9},
                "工作日 09:00–09:30、13:30–14:05（覆盖 90 %）", "workdays 09:00-09:30", 0.85)
    d = VW.department_statements([a, b], dept_of, "mail", when_of=when_of)[0]
    assert seen == [(["p:mail:0:7@1.0", "p:mail:0:8@1.0"], ["10.168.7.121", "192.168.1.21", "192.168.1.23"])]
    assert d["evidence"]["when"]["workday"] == [[540, 570], [810, 845]]
    assert d["confidence"] == 0.85 and d["evidence"]["facet_conf"]["when"] == 0.85
    assert "09:00–09:30" in d["text_zh"]
    # no reservoir fit: the union of the pieces' windows
    d2 = VW.department_statements([a, b], dept_of, "mail", when_of=lambda p, m: None)[0]
    assert d2["evidence"]["when"]["workday"] == [[0.0, 942.0], [1410.0, 1440.0]]


def test_before_any_prior_the_first_test_pass_rate_of_stated_nodes_is_the_base_rate():
    """Pack O seed 0, day 7, 22:15: P04 had issued no prior yet (its first
    pooled fit runs at the next daily step). The base rate is what P04 will
    issue an untested statement: the share of STATED nodes that passed their
    first held-out test, (passes + 0.5) / (n + 1) - not the pooled tests of
    every node: a tree's root over several actions is never stated and fails
    most tests (pack O seed 1, day 7: the pooled record, 0.62, stated nine
    statements that all held)."""
    from types import SimpleNamespace as NS

    class HR:
        def __init__(self, ft):
            self.ft = ft

        def tests(self, t=None):
            return (1.0, 1.0) if self.ft else (0.0, 5.0)

        def first_test(self):
            return self.ft
    tree = NS(root=1, nodes={1: _N(hold=HR(False)), 2: _N(hold=HR(True)), 3: _N(hold=HR(True)),
                             4: _N(hold=HR(False)), 5: _N()})
    c = NS(ptm=NS(kinds={0: tree}), _stated={2, 3, 4, 5})
    assert abs(VW.base_hold(c) - (2.0 + 0.5) / (3.0 + 1.0)) < 1e-12
    c2 = NS(ptm=NS(kinds={0: tree}))                    # no stated set: every node but the root
    assert abs(VW.base_hold(c2) - 2.5 / 4.0) < 1e-12


def test_pooled_reservoirs_fit_the_departments_windows():
    """The department's windows are fitted on its pieces' own arrivals (each
    piece's node keeps its members' minutes): pack O seed 1, days 15-21, the
    .121 node's own fit was a day-long low-coverage window and the 192.168
    part's had only the afternoon; pooled, both of GA's mail windows."""
    from app.engines.behavior.lib import psketch as PS
    from app.engines.behavior.lib import pwindows as PW
    T0 = 1_757_000_000.0
    tz = 8 * 3600.0
    ra, rb = PS.WeightedReservoir(256, PS.H_M, 1), PS.WeightedReservoir(256, PS.H_M, 2)
    rng = __import__("random").Random(5)
    d = 0
    for day in range(16):
        if (day % 7) in (5, 6):
            continue
        d += 1
        base = T0 + day * 86400.0
        for ip, res in (("192.168.1.21", ra), ("192.168.1.23", ra), ("10.168.7.121", rb)):
            for lo, hi in ((565, 583), (810, 843)):
                m = rng.uniform(lo, hi)
                ts = base - tz + m * 60.0
                res.offer((0, m, ip), 1.0, ts)
    pooled = VW._PooledWhen(ra.items() + rb.items())
    pw = PW.part_when(pooled, ["192.168.1.21", "192.168.1.23", "10.168.7.121"], tz)
    assert pw is not None
    wd = pw["workday"]
    assert any(a <= 570 and b >= 578 for a, b in wd) and any(a <= 815 and b >= 838 for a, b in wd), wd


def test_a_piece_drifting_only_in_its_own_windows_does_not_hold_back_the_refit_department():
    """Pack O seed 1, days 19-21: the .121 piece was 'evolving' (its own window
    drifted) and the department statement, which states its refitted
    windows, inherited 'evolving' and left recall."""
    a = _st(7, ["192.168.1.21", "192.168.1.23"], [[810, 845]], (2000.0, 60000.0), 0.9, [SIB_A])
    b = _st(8, ["10.168.7.121"], [[808, 834]], (3000.0, 70000.0), 0.83, [SIB_B], state="evolving")
    b["evidence"]["when"]["drift"] = {"workday": {"k": 0, "n": 3}}
    b["evidence"]["facet_conf"] = {"who": 0.97, "when": 0.83}
    b["evidence"]["conf_rest"] = 0.97              # its statement without its own windows
    refit = ({"workday": [[564, 571], [811, 844]], "nonworkday": [], "coverage": 0.8, "confidence": 0.8},
             "工作日 09:24–09:31、13:31–14:04", "", 0.9)
    d = VW.department_statements([a, b], dept_of, "mail", when_of=lambda p, m: refit)[0]
    assert d["state"] == "confirmed" and abs(d["confidence"] - 0.9) < 1e-12
    # without the refit the department states the pieces' windows: their drift counts
    d2 = VW.department_statements([a, b], dept_of, "mail")[0]
    assert d2["state"] == "evolving" and d2["confidence"] == 0.83
    # a piece evolving for another reason (no window drift) keeps the department evolving
    b2 = dict(b, evidence=dict(b["evidence"], when={"workday": [[808, 834]], "coverage": 0.8}))
    assert VW.department_statements([a, b2], dept_of, "mail", when_of=lambda p, m: refit)[0]["state"] == "evolving"


def test_a_pieces_nominal_content_facet_does_not_set_the_department_confidence():
    """Pack O seed 1, days 15-21 (round-9 development run): 10.168.7.121's
    mail piece had no held-out test of its own; its statement stated 0.96 (its
    kind's prior) while its content facet carried the nominal lower bound of
    its 12 rows (0.05-0.53). The refitted department statement took the facet
    and stated 0.05-0.53 - and held on every day. A piece contributes its
    statement's confidence without its own windows (evidence conf_rest)."""
    a = _st(7, ["192.168.1.21", "192.168.1.23"], [[810, 845]], (2000.0, 60000.0), 0.97, [SIB_A])
    a["evidence"]["conf_rest"] = 0.97
    b = _st(8, ["10.168.7.121"], [[0, 942]], (3000.0, 70000.0), 0.96, [SIB_B])
    b["evidence"]["facet_conf"] = {"who": 0.96, "when": 0.5, "content": 0.0544}
    b["evidence"]["conf_rest"] = 0.96
    refit = ({"workday": [[564, 571], [811, 844]], "nonworkday": [], "coverage": 0.8, "confidence": 0.8},
             "工作日 09:24–09:31、13:31–14:04", "", 0.98)
    d = VW.department_statements([a, b], dept_of, "mail", when_of=lambda p, m: refit)[0]
    assert abs(d["confidence"] - 0.96) < 1e-12
    # without conf_rest (an older piece): the piece's statement confidence
    b["evidence"].pop("conf_rest")
    d = VW.department_statements([a, b], dept_of, "mail", when_of=lambda p, m: refit)[0]
    assert abs(d["confidence"] - 0.96) < 1e-12
