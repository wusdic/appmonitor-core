"""Org generator (progressive.md §11): the requirement's example, determinism,
truth consistency, drifts, anomalies, real-world perturbations."""
from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter, defaultdict
from urllib.parse import parse_qs

import numpy as np
import pytest

from app.eval import packs as P
from app.eval import truth as T
from app.pipeline import orggen as G
from app.pipeline.generator import TrafficGenerator

KB = 1024


def _pack(name="O", n_days=None, **kw):
    if kw or n_days:
        spec = G.build_org(name, n_days=n_days or 21, **kw)
        return P._org_pack(name, spec, "test")
    return P.get_pack(name)


def _run(pack, seed=0, dt=900.0, n=None):
    g = TrafficGenerator(seed=seed, pack=pack)
    obs = []
    for _ in range(n if n is not None else int(round((pack.end_epoch - pack.start_epoch) / dt))):
        obs.extend(g.step(dt))
    return g, obs


def _events(obs):
    """Per-event rows (ev_sample rows or event-mode records) with their record."""
    for o in obs:
        rows = o.extra.get("ev_sample")
        if rows is not None:
            for r in rows:
                yield o, o.ts + r["o"], r.get("up"), r.get("l7"), r.get("meta")
        elif "count" not in o.extra:
            yield o, o.ts, o.bytes_up, o.extra.get("l7"), o.extra.get("meta")


@pytest.fixture(scope="module")
def full_o():
    """Pack O, seed 0, all 21 days (shared by the truth-consistency tests)."""
    pack = P.get_pack("O")
    g, obs = _run(pack)
    return pack, g, obs


def test_example_addresses_and_packs_registered():
    pack = P.get_pack("O")
    assert pack.org is not None and pack.ev_sample and pack.n_warmup_phases == 0
    sysmap = {s.id: s for s in pack.org.systems}
    assert sysmap["oa"].addr == "192.168.100.100:8080"
    ga = next(d for d in pack.org.departments if d.code == "GA")
    assert ga.name == "综合部" and ga.ips == ["192.168.1.21", "192.168.1.23", "10.168.7.121"]
    assert ga.usernames == {"192.168.1.21": "jack", "192.168.1.23": "rose", "10.168.7.121": "mike"}
    assert set(P.PACKS) == {"A", "B", "C", "D", "E", "smoke"}          # existing table untouched
    for n in ("O", "O60", "O-real", "o_real", "O-red", "O-servers", "O-servers-20",
              "O-scale-5k-60", "O-real-R3", "O-real-R1p"):
        assert P.get_pack(n).org is not None
    assert P.get_pack("A").org is None and not P.get_pack("A").ev_sample


def test_determinism_and_seed_sensitivity():
    pack = _pack("O", n_days=1)
    _, a = _run(pack, 3)
    _, b = _run(_pack("O", n_days=1), 3)
    _, c = _run(_pack("O", n_days=1), 4)
    key = lambda o: (o.ts, o.system, o.entity, o.http_path, o.bytes_up, o.extra.get("count"))
    assert [key(o) for o in a] == [key(o) for o in b]
    assert [o.extra.get("ev_sample") for o in a] == [o.extra.get("ev_sample") for o in b]
    assert [key(o) for o in a] != [key(o) for o in c]


def test_traffic_independent_of_tick_length():
    """The same day at 900 s aggregated and 60 s event mode: identical events."""
    pack = _pack("O", n_days=1)
    _, agg = _run(pack, 1, 900.0)
    _, ev = _run(_pack("O", n_days=1), 1, 60.0)
    def counts(obs):
        c = Counter()
        for o in obs:
            c[(o.system, o.entity, o.http_method, o.http_path, o.tls_sni)] += int(o.extra.get("count", 1))
        return c
    assert counts(agg) == counts(ev)
    assert all("l7" in o.extra for o in ev if o.app_proto == "http")
    assert sum(int(o.extra.get("count", 1)) for o in agg) == len(ev)


def test_ev_sample_contract():
    pack = _pack("O", n_days=1)
    _, obs = _run(pack, 0)
    for o in obs:
        rows = o.extra["ev_sample"]
        w = o.extra["count"]
        assert len(rows) == min(w, G.EV_SAMPLE_MAX)
        offs = [r["o"] for r in rows]
        assert offs == sorted(offs) and all(0.0 <= x < 900.0 + 3600 for x in offs)
        if w <= G.EV_SAMPLE_MAX:
            assert sum(r["up"] for r in rows) == pytest.approx(o.extra["bytes_up_total"], abs=w)
        if o.app_proto == "tls":
            assert all("l7" not in r for r in rows) and not o.http_method
        if o.system == "oa" and o.http_method:
            assert all(r["l7"]["headers"]["host"] == "oa.corp.local" for r in rows)


def test_ga_login_example_facts(full_o):
    """The requirement's sentence, measured on the generated traffic."""
    pack, g, obs = full_o
    ips = {"192.168.1.21": "jack", "192.168.1.23": "rose", "10.168.7.121": "mike"}
    sizes, minutes, names = [], [], defaultdict(set)
    for o, ts, up, l7, meta in _events(obs):
        if o.system != "oa" or o.http_path != "/login" or o.entity not in ips or l7 is None:
            continue
        d = g.org.day_of(ts)
        if d >= 12 or o.http_method != "POST":
            continue                                  # before D1 / D2; A2-A4 are later
        body = l7["body"]
        kv = parse_qs(body)
        names[o.entity].add(kv["username"][0])
        sizes.append(l7["body_len"])
        ld = g.clock.local(ts)
        minutes.append(ld.hour * 60 + ld.minute + ld.second / 60)
        assert "username=" in body and len(kv["username"][0]) <= 10
    assert {k: v for k, v in names.items()} == {k: {v} for k, v in ips.items()}
    s = np.asarray(sizes)
    assert s.min() >= 0.5 * KB and s.max() <= 3 * KB
    assert np.mean((s >= KB) & (s <= 2 * KB)) >= 0.8          # 27 logins; the spec itself below
    spec = next(a for a in pack.org.activities if a.name == "GA.oa.login").steps[0].body
    r = np.random.default_rng(5)
    big = np.asarray([len(G.render_body(spec, r, {"username": "jack"})[0].encode()) for _ in range(4000)])
    assert big.min() >= 0.5 * KB and big.max() <= 3 * KB
    assert 0.88 <= np.mean((big >= KB) & (big <= 2 * KB)) <= 0.92
    m = np.asarray(minutes)
    assert m.min() >= 540 and m.max() <= 561
    assert o.peer == "192.168.100.100" and o.dst_port == 8080


def test_truth_consistency(full_o):
    """Every benign event conforms to the truth row it was generated under."""
    pack, g, obs = full_o
    pt = {r["tid"]: r for r in g.ptruth["pattern_truth"]}
    # re-plan every day (planning is deterministic and independent of emission)
    g2 = TrafficGenerator(seed=0, pack=P.get_pack("O"))
    n = 0
    win_in = defaultdict(list)
    for d in range(1, 22):
        for e in g2.org._plan_day(d):
            if e.tid is None:
                continue
            row = pt[e.tid]
            n += 1
            assert row["valid_from_day"] <= d < row["valid_to_day"], e.tid
            assert e.system in row["systems"] or row["system"] == e.system
            if row["channel"] == "http":
                path_rx = "^" + re.sub(r"\\\{\w+\\\}", r"\\d+", re.escape(row["route"])) + "$"
                assert re.match(path_rx, e.path) and e.method == row["method"], (e.path, row["route"])
            who = row["who"]
            if who["level"] == "ip":
                assert e.src in who["value"]
            elif who["level"] == "grp":
                assert e.src in who["members"]
            else:
                assert any(ipaddress.ip_address(e.src) in ipaddress.ip_network(c) for c, _ in who["value"])
            ld = g2.clock.local(e.ts)
            dt_ = "workday" if g2.clock.day_kind(ld.date())[0] else "nonworkday"
            minute = ld.hour * 60 + ld.minute + ld.second / 60.0
            # every arrival inside the step's support; its windows hold the
            # central 99 % of the step's arrival law (round 3, §11.5)
            assert any(a <= minute <= b for a, b in row["gen"]["support"][dt_]), (e.tid, minute)
            win_in[e.tid].append(any(a <= minute <= b for a, b in row["windows"][dt_]))
            c = row["content"]
            if "body.len" in c and e.l7 is not None:
                lo, hi = c["body.len"]["range"]
                assert lo - 2 <= e.l7["body_len"] <= hi + 2, (e.tid, e.l7["body_len"])
            if "body.keys" in c and e.l7 is not None:
                keys = set(parse_qs(e.l7["body"], keep_blank_values=True)) if e.l7["body_type"] == "form" \
                    else set(re.findall(r'"(\w+)":', e.l7["body"][:200]))
                if e.l7["body_type"] == "form" and not e.l7["body_trunc"]:
                    assert keys == set(c["body.keys"]["required_keys"])
            for attr, table in row["bindings"].items():
                key = attr.split(".")[-1]
                v = parse_qs(e.l7["body"])[key][0]
                want = table[e.src]
                assert (v in want) if isinstance(want, list) else v == want
            for attr, tc in c.items():
                if attr.startswith("body.kv.") and tc.get("grammar") and e.l7 and \
                        e.l7["body_type"] == "form":
                    v = parse_qs(e.l7["body"], keep_blank_values=True)[attr[8:]][0]
                    assert re.fullmatch(tc["grammar"]["regex"], v), (attr, v)
    assert n > 100000
    for tid, hits in win_in.items():
        assert np.mean(hits) >= 0.99 - 3.0 * math.sqrt(0.01 * 0.99 / len(hits)) - 1.0 / len(hits), \
            (tid, np.mean(hits), len(hits))
    # opportunities recorded while emitting == planned benign events
    tot = sum(c for per in g.ptruth["opportunities"].values() for day in per.values()
              for c in day.values())
    assert tot == g.org.stats["benign"]


def test_drifts(full_o):
    pack, g, obs = full_o
    rows = {r["tid"]: r for r in g.ptruth["pattern_truth"]}
    assert rows["GA.oa.login#0"]["windows"]["workday"] == [[540, 561]]
    assert rows["GA.oa.login#0@1"]["windows"]["workday"] == [[510, 531]]
    assert rows["GA.oa.login#0@2"]["bindings"]["body.kv.username"]["10.168.7.121"] == "mike.w"
    assert rows["GA.oa.approvals#2@1"]["route"] == "/flow/{id}/approve"
    assert rows["GA.oa.approvals#2"]["valid_to_day"] == 14
    tr = {r["scenario_id"]: r for r in g.truth}
    assert tr["D1"]["tids_new"] and "GA.oa.login#0" in tr["D1"]["tids_old"]
    # DHCP: pool IPs change with the lease (24 h, 12 h from day 15), inside the pool
    pool = ipaddress.ip_network("10.50.0.0/22")
    a = g.org.actors["DEV#000"]
    ips = [g.org.ip_at(a, g.org.day_start(d) + 3600) for d in range(1, 22)]
    assert all(ipaddress.ip_address(x) in pool for x in ips) and len(set(ips)) >= 18
    d15 = g.org.day_start(16)
    assert g.org.ip_at(a, d15 + 3600) != g.org.ip_at(a, d15 + 13 * 3600)
    assert g.org.ip_at(a, g.org.day_start(3) + 3600) == g.org.ip_at(a, g.org.day_start(3) + 13 * 3600)
    # D4: portal population grows inside days 12-18
    per_day = {d: len({ip for ip in g.ptruth["who_log"]["portal"].get(g.org.date_of(d).isoformat(), {})
                       if ip.startswith(("10.6", "172.16"))}) for d in range(1, 22)}
    assert np.mean([per_day[d] for d in range(12, 19)]) > 1.2 * np.mean([per_day[d] for d in range(1, 12)])


def test_anomaly_rows_and_events(full_o):
    pack, g, obs = full_o
    tr = {r["scenario_id"]: r for r in g.truth}
    assert {f"A{i}" for i in range(1, 11)} <= set(tr)
    assert not T.validate([r for r in g.truth if r["label"] == "malicious"
                           and r["scenario_id"] in ("A1", "A3", "A6", "A7", "A10")])
    for sid in ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"):
        r = tr[sid]
        assert r["expected_types"] and r["required_severity"] and r["t_first"] >= r["t_start"]
    evs = defaultdict(list)
    for d in range(1, 22):
        for e in g.org._anom_events.get(d, []):
            evs[e.anomaly].append(e)
    a1 = evs["A1"]
    assert {e.system for e in a1} == {"finance"} and {e.src for e in a1} == {"192.168.1.23"}
    assert [e.method for e in a1] == ["GET", "POST"]
    a2 = evs["A2"]
    assert all("username=rose" in e.l7["body"] for e in a2 if e.method == "POST")
    assert len({g.org.day_of(e.ts) for e in a2}) == 5 and all(e.src == "192.168.1.21" for e in a2)
    a3 = [e for e in evs["A3"] if e.method == "POST"][0]
    assert parse_qs(a3.l7["body"])["username"][0] == "admin' OR '1'='1" and a3.l7["body_len"] >= 12 * KB
    assert a3.meta["waf.score"] >= 12
    a4 = evs["A4"][0]
    assert g.clock.local(a4.ts).hour == 3 and "username=mike.w" in a4.l7["body"]
    a5 = evs["A5"]
    assert [e.path for e in a5] == ["/report/generate"]
    d19 = g.org._plan_day(19)
    assert not [e for e in d19 if e.src == "192.168.1.23" and e.path == "/report/form"]
    a8 = evs["A8"]
    assert a8[0].src == "192.168.2.99" and "username=lucy" in a8[0].l7["body"]
    t8 = a8[0].ts
    assert any(e.src == "192.168.2.10" and "/fin/approval" in e.path and abs(e.ts - t8) < 3600
               for e in d19)
    assert tr["A8"]["nominal_day"] == 20 and g.org.day_of(t8) == 19
    assert len({g.org.day_of(e.ts) for e in evs["A9"]}) == 11
    a7 = evs["A7"]
    assert len(a7) == 400 and np.mean([e.status == 401 for e in a7]) > 0.8
    assert len({e.src for e in evs["A10"]}) >= 40
    assert g.org.stats["anomalous"] == sum(len(v) for v in evs.values())


def test_attribute_schedule_and_truth():
    pack = _pack("O-scale", n_days=4, n_meta=60)
    g, obs = _run(pack, 0)
    at = {a["name"]: a for a in g.ptruth["attr_truth"]}
    syn = [a for a in at.values() if a["name"].startswith("meta.f")]
    assert len(syn) == 60
    cls = Counter(a["cls"] for a in syn)
    assert cls["informative"] == 12 and cls["constant"] == 8 and cls["noise"] == 40
    first = min(a["appears"] for a in syn)
    assert g.org.day_of(first) == 3
    seen = set()
    for o, ts, up, l7, meta in _events(obs):
        for k in meta or {}:
            seen.add(k)
            if k.startswith("f"):
                assert g.org.day_of(ts) >= 3
    assert {a["name"][5:] for a in syn} <= seen
    # by-department attribute is constant within a department
    inf = next(a["name"][5:] for a in syn if a["cls"] == "informative" and a["by"] == "dept")
    vals = defaultdict(set)
    for o, ts, up, l7, meta in _events(obs):
        if meta and inf in meta and o.system == "oa" and o.entity.startswith("192.168.1."):
            vals["GA"].add(meta[inf])
    assert len(vals["GA"]) == 1


def test_real_world_perturbations():
    pack = P.pack_o_real(["R1", "R9", "R12", "R3", "R4", "R5", "R2"], name="O-real-t")
    pack.org.n_days = 11
    g = TrafficGenerator(seed=0, pack=pack)
    obs = []
    for _ in range(11 * 96):
        obs.extend(g.step(900.0))
    oa8 = [o for o in obs if o.system == "oa" and g.org.day_of(o.ts) >= 8 and o.http_method]
    assert oa8 and all(o.entity == "192.168.100.99" for o in oa8)
    xff = {r["l7"]["headers"].get("x-forwarded-for") for o in oa8 for r in o.extra["ev_sample"]}
    assert "192.168.1.21" in xff
    assert "192.168.100.99/32" in pack.config["progressive"]["trusted_proxies"]
    assert any(o.system == "oa-r2" for o in obs if g.org.day_of(o.ts) >= 10)
    assert not any(o.system == "oa-r2" for o in obs if g.org.day_of(o.ts) < 10)
    portal = [o for o in obs if o.system == "portal"]
    assert portal and all(o.extra.get("sample_rate") == 4.0 for o in portal)
    assert g.org.stats["dropped"] > 0
    fin = {o.entity for o in obs if o.system == "finance" and g.org.day_of(o.ts) >= 9}
    assert "192.168.2.51" in fin and "192.168.2.11" not in fin
    nat = [r for o in obs if o.entity == "192.168.30.1" for r in o.extra["ev_sample"]
           if "body" in (r.get("l7") or {})]
    assert len({parse_qs(r["l7"]["body"])["username"][0] for r in nat if "username" in r["l7"]["body"]}) >= 8
    rows = {r["tid"]: r for r in g.ptruth["pattern_truth"]}
    assert rows["GAT.oa.login#0"]["bindings"]["body.kv.username"] == {"192.168.1.40": ["jack2", "rose2"]}
    tr = {r["scenario_id"] for r in g.truth}
    assert {"R1", "R2", "R3", "R4", "R5", "R9", "R12"} <= tr


def test_red_team_differs():
    red = P.get_pack("O-red")
    o = P.get_pack("O")
    ga_r = next(d for d in red.org.departments if d.code == "GA")
    assert set(ga_r.ips).isdisjoint({"192.168.1.21", "192.168.1.23", "10.168.7.121"})
    assert "ip_classes" not in red.config and "ip_classes" in o.config
    assert sum(1 for a in red.org.attr_schedule if a.cls == "independent") == 20


def test_servers_org_shape():
    spec = G.build_servers_org()
    assert len(spec.systems) == 300 and len(spec.families) == 12
    assert all(len(v) == 20 for v in spec.families.values())
    for n in (20, 100):
        assert len(G.build_servers_org(n_systems=n).systems) == n
    pack = P.get_pack("O-servers-20")
    pack.org.n_days = 2
    g, obs = _run(pack, 0, n=2 * 96)
    idle = set(pack.org.config["idle_systems"])
    assert not [o for o in obs if o.system in idle and g.org.day_of(o.ts) > 1]


def test_grammar_helpers():
    assert G.grammar_of(["jack", "rose", "mike"])["regex"] == "[a-z]{4}"
    g = G.grammar_of(["jack", "rose", "mike.w"])
    assert g["regex"] == "[a-z.]{4,6}"
    r = np.random.default_rng(0)
    c = G.contrast_values(g, r, 500)
    assert not any(re.fullmatch(g["regex"], x) for x in c)
    s = G.SizeSpec([(0.9, 1024, 2048), (0.05, 512, 1024), (0.05, 2048, 3072)], (512, 3072))
    assert s.quantile(0.05) == pytest.approx(1024, abs=1) and s.quantile(0.95) == pytest.approx(2048, abs=1)
    x = np.asarray([s.sample(r) for _ in range(4000)])
    assert x.min() >= 512 and x.max() <= 3072


@pytest.mark.parametrize("builder", [lambda: G.build_org("O", real=P.R_ITEMS),
                                     lambda: G.build_org("O-red", red=True),
                                     lambda: G.build_servers_org()])
def test_body_specs_fit_their_size(builder):
    """Every body's fields fit the size it is padded to, so the truth's size
    band/range (computed from the SizeSpec) is what the traffic carries."""
    spec = builder()
    r = np.random.default_rng(0)
    for act in spec.activities:
        for st in act.steps:
            b = st.body
            if b is None or b.size is None:
                continue
            lo, hi = b.size.support()
            for _ in range(60):
                body, _ = G.render_body(b, r, {"username": "abcdefgh", "dept": "GA", "aid": "x"})
                assert len(body.encode()) <= hi + 1, (act.name, st.route_fmt, len(body.encode()))


def test_attribute_appears_is_first_observable_time():
    """PG7 truth: an attribute 'appears' when it first reaches a capture record
    (in aggregated mode only ev_sample rows carry l7 / meta), not when the
    generator first planned it — an event left out of ev_sample is invisible."""
    pack = _pack("O-scale", n_days=4, n_meta=60)
    g, obs = _run(pack, 0)
    first = {}
    for o, ts, up, l7, meta in _events(obs):
        names = [f"hdr.{h}" for h in ((l7 or {}).get("headers") or {})] + [f"meta.{m}" for m in (meta or {})]
        for k in names:
            first[k] = min(first.get(k, math.inf), ts)
    for a in g.ptruth["attr_truth"]:
        if a["name"] in first:
            assert a["appears"] == pytest.approx(first[a["name"]], abs=1e-6), a["name"]


def test_payload_string_attributes_are_typed_text_in_the_truth():
    """PG7 truth semantics (progressive.md §5.4.5, §16.2 A2): a string value in
    a payload namespace (hdr.*) is `text` (shape hierarchy, P07 grammar and
    closed set); a categorical truth for a 2-valued header contradicted the
    hierarchy the spec assigns to it."""
    from app.pipeline.orggen import build_org
    spec = build_org("O")
    for a in spec.attr_schedule:
        if a.where == "headers":
            assert a.type == "text", (a.name, a.type)


def test_org_packs_default_to_a_registry_that_fits_in_memory():
    """Pack O's own default must be runnable: `full+progressive` (B01-B30 +
    P-core) held 2.26 GB by day 4 and was OOM-killed at 7.1 GB per run
    (progressive.md §16.2 M25), so every measured run had to override it with
    --registry progressive_decision. The org packs now default to it; the
    scaling / server packs keep progressive_only."""
    for n in ("O", "O60", "O-red", "O-real", "O-real-R3"):
        assert P.get_pack(n).registry_mode == "progressive_decision", n
    assert P.get_pack("O-servers-20").registry_mode == "progressive_only"
    assert P.get_pack("O-scale-500-0").registry_mode == "progressive_only"


def _ks_to_law(x, q):
    """Kolmogorov distance between a sample and a law given by its quantile
    function at equally spaced probabilities."""
    x = np.sort(np.asarray(x, dtype=float))
    F = np.interp(x, np.asarray(q, dtype=float), np.linspace(0.0, 1.0, len(q)))
    emp = np.arange(1, len(x) + 1) / len(x)
    return float(np.max(np.maximum(np.abs(F - emp), np.abs(F - (emp - 1.0 / len(x))))))


def test_step_truth_is_the_steps_own_arrival_law():
    """Round 3 (truth fix): a step's truth windows and held-out minutes follow
    the step's own arrival law (session start + think times, repeats,
    optional steps), not the activity's window widened by the think-time
    sums with minutes drawn uniformly in it. GA mail records 2-8 of a session
    were drawn in a 09:40-09:47 tail the generator hardly reaches (KS 0.23)."""
    from app.eval import pmetrics as M
    spec = G.build_org("O", n_days=3)
    g = G.OrgGenerator(spec, seed=0)
    r = np.random.default_rng(7)
    for name in ("GA.mail", "PUB.portal.visit"):
        act = next(a for a in spec.activities if a.name == name)
        e = g.eff(act, 1)
        w = e.when[0]
        actor = g.actors[g.members[act.dept][0]] if act.dept != "PUB" else \
            next(a for a in g.actors.values() if a.dept == "PUB")
        arr = defaultdict(list)
        for _ in range(2500):
            if w.arrival == "normal":
                m = float(np.clip(r.normal(0.5 * (w.m0 + w.m1), (w.m1 - w.m0) / 4), w.m0, w.m1))
            else:
                m = float(r.uniform(w.m0, w.m1))
            sysid = act.system if isinstance(act.system, str) else act.system[0]
            for ev in g._session(act, e, actor, g._minute_ts(1, m), r, 1, sysid):
                k = int(ev.tid.split("#")[1].split("@")[0])
                arr[k].append((ev.ts - g.day_start(1)) / 60.0)
        rows = {row["step"]: row for row in g.ptruth()["pattern_truth"]
                if row["activity"] == name and row["valid_from_day"] == 1}
        for k, x in arr.items():
            row = rows[k]
            dt = "workday"
            # the truth row's held-out minutes are the generator's arrivals
            # (the sessions above start in the first window spec only)
            a0, b0 = row["windows"][dt][0]
            hold = [h["minute"] for h in M.holdout_events([row], np.random.default_rng(k), 6000)
                    if h["daytype"] == dt and h["minute"] <= b0 + 30]
            qh = np.quantile(hold, np.linspace(0, 1, 129))
            tol = max(0.05, 1.63 / math.sqrt(len(x)) + 0.01)       # 1 % KS level + quantile noise
            assert _ks_to_law(x, qh) < tol, (name, k, _ks_to_law(x, qh))
            # the stated window holds >= 98 % of the arrivals
            inside = np.mean([any(a <= v <= b for a, b in row["windows"][dt]) for v in x])
            assert inside >= 0.98, (name, k, inside, row["windows"][dt])
    # the GA login (step 0, one record) keeps the activity's window exactly
    login = next(row for row in g.ptruth()["pattern_truth"] if row["tid"] == "GA.oa.login#0")
    assert login["windows"]["workday"] == [[G._hm("09:00"), G._hm("09:21")]]


def test_tls_truth_states_the_observed_upstream_bytes():
    """Evaluator round 3: a TLS row's net.bytes_up is the step's payload plus the
    TLS framing; the truth's band / range must describe that observed quantity.
    The truth stated the bare payload, so the learned mail and git bands were
    exactly truth + 300 B and failed PG1 content on every seed."""
    spec = G.build_org("O", n_days=3)
    g = G.OrgGenerator(spec, seed=0)
    rows = {r["tid"]: r for r in g.ptruth()["pattern_truth"]}
    ups = defaultdict(list)
    for d in (1, 2):
        for e in g._plan_day(d):
            if e.ch == "t" and e.tid is not None and "net.bytes_up" in rows[e.tid]["content"]:
                ups[e.tid].append(e.up)
    assert ups, "no TLS step with a bytes_up truth"
    for tid, x in ups.items():
        c = rows[tid]["content"]["net.bytes_up"]
        lo, hi = c["range"]
        assert all(lo - 1 <= v <= hi + 1 for v in x), (tid, lo, hi, min(x), max(x))
        if len(x) >= 100:
            # the stated lower edge is reached: the smallest observed value is
            # close to it (the bare payload's edge lies 300 B below every row)
            assert min(x) - lo <= 0.2 * lo, (tid, lo, min(x))
            b0, b1 = c["band90"]
            inside = np.mean([b0 - 1 <= v <= b1 + 1 for v in x])
            assert inside >= 0.75, (tid, inside, c["band90"])
