"""Generator v2 (docs/lib3/generator.md): determinism, persona individuation,
twins, machine clocks, human rhythm, calendar / 调休 / DST, aggregated mode
and the backward-compatible v1 step()."""
import ast
import datetime as dt
import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pytest

from app.pipeline import generator as G
from app.pipeline.generator import (BASE_KEYS, ORG_STANDARD, SMOKE_KEYS, TWINS, Clock,
                                    TrafficGenerator, build_persona)
from app.eval.packs import get_pack

TZ = "Asia/Shanghai"


@dataclass
class MiniPack:
    """A Pack-like object with only what the generator reads."""
    name: str = "t"
    tz: str = TZ
    calendar: Dict[str, List[str]] = field(default_factory=lambda: {"holidays": [],
                                                                    "makeup_workdays": []})
    start_epoch: float = 0.0
    population: List[str] = field(default_factory=list)
    scenarios: List[Any] = field(default_factory=list)


def local_epoch(d: dt.date, hh: float = 0.0, tz: str = TZ) -> float:
    return Clock(tz).epoch(d, hh)


def run(gen: TrafficGenerator, n: int, dt_s: float, **kw) -> List[Any]:
    out = []
    for _ in range(n):
        out.extend(gen.step(dt_s, **kw))
    return out


def w(o) -> int:
    return int((o.extra or {}).get("count", 1))


def sig(o):
    return (o.ts, o.system, o.entity, o.http_method, o.http_path, o.http_status, o.bytes_up,
            o.bytes_down, o.user_agent, o.ja3, o.dns_qname, o.rtt_ms, repr(sorted((o.extra or {})
                                                                               .items())))


MONDAY = dt.date(2025, 3, 10)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_deterministic_and_seeded():
    def obs(seed):
        g = TrafficGenerator(seed=seed, pack=get_pack("mini"))
        return [sig(o) for o in run(g, 30, 3600.0)]
    a, b, c = obs(3), obs(3), obs(4)
    assert a == b and len(a) > 100
    assert a != c


def test_persona_params_from_crc32_only():
    # the same individual in every run, seed and population
    p1 = build_persona("erp-prod|10.20.1.13").params
    g = TrafficGenerator(seed=99, pack=MiniPack(population=["erp-prod|10.20.1.13"]))
    assert g.personas["erp-prod|10.20.1.13"].params == p1
    assert build_persona("erp-prod|10.20.1.12").params != p1
    calls = [n.func.id for n in ast.walk(ast.parse(inspect.getsource(G)))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "hash" not in calls                                # never Python hash()


# --------------------------------------------------------------------------- #
# Persona v2
# --------------------------------------------------------------------------- #
def _human(key):
    return build_persona(key).models[0]


def _jsd(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def test_individuated_personas_and_twins():
    keys = [k for k in BASE_KEYS if build_persona(k).archetype == "interactive"
            and k not in TWINS and k.startswith("oa-portal")]
    hs = [_human(k) for k in keys]
    for h in hs:
        assert 30 <= h.K <= 40
    pairs = [_jsd(a.visit, b.visit) for i, a in enumerate(hs) for b in hs[i + 1:]]
    assert min(pairs) > 0.15                         # sparse Dirichlet(0.4): distinct mixes
    starts = [h.work_start for h in hs]
    assert np.std(starts) > 0.2 and all(6.5 <= s <= 11 for s in starts)
    a, b = _human(TWINS[0]), _human(TWINS[1])
    assert _jsd(a.visit, b.visit) < 0.25 * min(pairs)  # twins are the confusable pair
    assert abs(a.work_start - b.work_start) < 0.3 and abs(a.work_end - b.work_end) < 0.4
    assert a.stack == b.stack


def test_persona_param_ranges_and_hr_paths():
    for k in BASE_KEYS:
        p = build_persona(k)
        if p.archetype not in ("interactive", "search"):
            continue
        h = p.models[0]
        assert 0.01 <= h.err <= 0.08 and h.think_mu == pytest.approx(math.log(8.0))
        salary = [t for t in h.vocab if "/hr/salary" in t.fmt]
        assert bool(salary) == (k == "erp-prod|10.20.1.21")
        toks = {t.fmt for t in h.vocab}
        if k.startswith("erp"):
            assert {"/fin/ledger", "/hr/leave", "/crm/lead", "/wh/stock",
                    "/orders/view/{id}"} <= toks
            assert any(f.startswith("/report/") for f in toks)
        assert set(p.params) >= {"work_start", "work_end", "path_pref", "persona_id",
                                 "volume_scale", "weekday_mask", "device", "id_range"}


def test_device_tuple_split_70_30():
    std = [G.build_human(f"erp-prod|10.99.{i // 250}.{i % 250}").stack is ORG_STANDARD
           for i in range(600)]
    assert 0.63 < np.mean(std) < 0.77
    uniq = {G.build_human(f"erp-prod|10.99.{i // 250}.{i % 250}").stack.ja3 for i in range(600)}
    assert len(uniq) > 100                          # the 30 % are unique stacks


def test_rtt_per_subnet():
    g = TrafficGenerator(seed=0, pack=MiniPack())
    a, b = g._net("erp-prod|10.20.1.11"), g._net("erp-prod|10.20.1.12")
    c = g._net("erp-prod|10.20.4.30")
    assert 2.7 <= a <= 50 and abs(a - b) / a < 0.25
    assert abs(math.log(a / c)) > 0 or a != c


# --------------------------------------------------------------------------- #
# Rhythm, idle ticks, calendar, DST
# --------------------------------------------------------------------------- #
HUMANS = ["erp-prod|10.20.1.11", "erp-prod|10.20.1.12", "erp-prod|10.20.1.13",
          "oa-portal|10.30.2.21", "oa-portal|10.30.2.22"]


def _day_counts(tz, cal, day, keys=HUMANS, dt_s=900.0):
    g = TrafficGenerator(seed=1, pack=MiniPack(tz=tz, calendar=cal, population=keys,
                                               start_epoch=local_epoch(day, 0.0, tz)))
    clk = Clock(tz, cal)
    hours = np.zeros(24)
    empty = 0
    for _ in range(int(86400 / dt_s)):
        o = g.step(dt_s)
        empty += not o
        for x in o:
            if x.app_proto == "http":
                hours[clk.local(x.ts).hour] += w(x)
    return hours, empty


def test_humans_idle_at_night_and_real_idle_ticks():
    hours, empty = _day_counts(TZ, None, MONDAY + dt.timedelta(days=1))
    assert hours[:6].sum() == 0 and hours[22:].sum() == 0
    assert hours[9:17].sum() > 200
    assert hours[12] < hours[10]                      # lunch dip
    assert empty >= 20                                # idle ticks are really empty


def test_calendar_weekend_holiday_and_makeup_saturday():
    cal = {"holidays": ["2025-03-12"], "makeup_workdays": ["2025-03-15"]}
    wd, _ = _day_counts(TZ, cal, dt.date(2025, 3, 11))         # Tue workday
    hol, _ = _day_counts(TZ, cal, dt.date(2025, 3, 12))        # Wed holiday
    sat, _ = _day_counts(TZ, cal, dt.date(2025, 3, 15))        # 调休 make-up Saturday
    sun, _ = _day_counts(TZ, cal, dt.date(2025, 3, 16))        # plain Sunday
    assert hol.sum() == 0 and sun.sum() == 0
    assert sat.sum() > 0.5 * wd.sum() > 0


def test_dst_work_window_is_local():
    tz = "Europe/Berlin"
    before, _ = _day_counts(tz, None, dt.date(2025, 3, 28))    # CET
    after, _ = _day_counts(tz, None, dt.date(2025, 3, 31))     # CEST
    first = lambda h: int(np.argmax(h > 0))                    # noqa: E731
    assert abs(first(before) - first(after)) <= 1
    assert before[:5].sum() == 0 and after[:5].sum() == 0
    # in UTC the same local window moved by one hour
    c = Clock(tz)
    a = c.epoch(dt.date(2025, 3, 28), 9.0) % 86400
    b = c.epoch(dt.date(2025, 3, 31), 9.0) % 86400
    assert a - b == 3600


# --------------------------------------------------------------------------- #
# Machine personas
# --------------------------------------------------------------------------- #
def test_machine_clocks():
    keys = ["erp-prod|10.20.9.9", "erp-prod|10.20.4.31", "erp-prod|10.20.4.30",
            "oa-portal|10.30.2.50"]
    g = TrafficGenerator(seed=2, pack=MiniPack(population=keys,
                                               start_epoch=local_epoch(MONDAY, 10.0)))
    obs = run(g, 120, 60.0)
    by = {k: [o for o in obs if f"{o.system}|{o.entity}" == k] for k in keys}
    health = sorted(o.ts for o in by["erp-prod|10.20.9.9"] if o.http_path == "/healthz")
    gaps = np.diff(health)
    assert len(health) >= 235 and np.all((gaps >= 28.9) & (gaps <= 31.1))
    ntp = [o for o in by["erp-prod|10.20.9.9"] if o.app_proto == "ntp"]
    assert len(ntp) >= 100 and all(o.dst_port == 123 for o in ntp)
    sync = sorted(o.ts for o in by["erp-prod|10.20.4.31"] if o.http_path == "/api/sync")
    polls = [t for i, t in enumerate(sync) if i == 0 or t - sync[i - 1] > 60]
    assert 22 <= len(polls) <= 26 and np.allclose(np.diff(polls), 300, atol=6)
    api = by["erp-prod|10.20.4.30"]
    assert api and {o.user_agent for o in api if o.app_proto == "http"} <= {
        "python-requests/2.31.0", "okhttp/4.12.0", "Go-http-client/1.1"}
    nat = [o for o in by["oa-portal|10.30.2.50"] if o.app_proto == "http"]
    assert len(g.personas["oa-portal|10.30.2.50"].models) == 2 and nat


def test_backup_windows():
    keys = ["erp-prod|10.20.9.5", "api-gateway|10.40.9.6"]
    g = TrafficGenerator(seed=0, pack=MiniPack(population=keys,
                                               start_epoch=local_epoch(MONDAY, 0.0)))
    clk = Clock(TZ)
    obs = [o for o in run(g, 24 * 4, 900.0, aggregated=False) if o.http_method == "PUT"]
    for k, h in (("erp-prod|10.20.9.5", 1.0), ("api-gateway|10.40.9.6", 2.0)):
        hs = [clk.hour(o.ts) for o in obs if f"{o.system}|{o.entity}" == k]
        assert len(hs) > 150
        assert h - 10 / 60 - 1e-6 <= min(hs) and max(hs) <= h + 50 / 60 + 1e-6


# --------------------------------------------------------------------------- #
# Aggregated mode
# --------------------------------------------------------------------------- #
def test_aggregated_equals_raw_events_for_the_same_draws():
    pk = get_pack("mini")
    ga = TrafficGenerator(seed=5, pack=pk)
    gr = TrafficGenerator(seed=5, pack=get_pack("mini"))
    ga.vt = gr.vt = local_epoch(MONDAY, 8.0)
    for _ in range(12):
        agg = ga.step(900.0)                              # dt >= 900: aggregated
        raw = gr.step(900.0, aggregated=False)
        t0 = ga.vt - 900.0
        assert sum(w(o) for o in agg) == len(raw)
        assert sum(o.extra["bytes_up_total"] for o in agg) == pytest.approx(
            sum(o.bytes_up for o in raw), rel=1e-3, abs=len(raw))
        assert len(agg) < len(raw) or not raw
        keys = set()
        for o in agg:
            ex = o.extra
            assert {"count", "bytes_up_total", "bytes_down_total", "ts_sample"} <= set(ex)
            assert 1 <= len(ex["ts_sample"]) <= min(64, ex["count"])
            assert t0 <= o.ts and all(o.ts + s < ga.vt for s in ex["ts_sample"])
            k = (o.entity, o.app_proto, o.http_method, o.http_host, o.http_path, o.dns_qname,
                 o.http_status, o.user_agent, o.ja3)
            assert k not in keys                        # one per (token, outcome, dest, stack)
            keys.add(k)
        for o in raw:
            assert t0 <= o.ts < ga.vt and not o.extra


def test_human_traffic_mix_does_not_depend_on_the_tick_length():
    """Sessions are a continuous-time process carried across ticks. They used
    to be laid out inside each tick, so at dt = 60 s every minute started a
    session: ~4x the login redirects and DNS lookups, ~2.5x the POST share and
    a median inter-request gap of ~50 s instead of ~9 s, relative to 900 s
    (pack E and the Runtime then compared two populations)."""
    keys = HUMANS

    def mix(dt_s):
        acc = {"http": 0, "post": 0, "r3xx": 0, "dns": 0}
        gaps: List[float] = []
        for seed in range(2):
            g = TrafficGenerator(seed=seed, pack=MiniPack(population=keys,
                                                          start_epoch=local_epoch(MONDAY, 7.0)))
            last: Dict[str, float] = {}
            for o in run(g, int(12 * 3600 / dt_s), dt_s, aggregated=False):
                if o.app_proto == "http":
                    acc["http"] += 1
                    acc["post"] += o.http_method != "GET"
                    acc["r3xx"] += 300 <= o.http_status < 400
                    if o.entity in last and o.ts > last[o.entity]:
                        gaps.append(o.ts - last[o.entity])
                    last[o.entity] = o.ts
                elif o.app_proto == "dns":
                    acc["dns"] += 1
        return ({k: acc[k] / acc["http"] for k in ("post", "r3xx", "dns")},
                float(np.median(gaps)))
    (f60, g60), (f900, g900) = mix(60.0), mix(900.0)
    for k in f60:
        assert 0.7 < f60[k] / f900[k] < 1.4, (k, f60[k], f900[k])
    assert 0.7 < g60 / g900 < 1.4


def test_machine_dns_does_not_depend_on_the_tick_length():
    """An API client's resolver clock was skipped on ticks without a poll
    (half its lookups lost at 60 s) and a backup host resolved once per tick
    of its window (~60 lookups per run at 60 s, 4 at 900 s)."""
    keys = ["api-gateway|10.40.4.51", "erp-prod|10.20.9.5"]

    def dns(dt_s):
        g = TrafficGenerator(seed=2, pack=MiniPack(population=keys,
                                                   start_epoch=local_epoch(MONDAY, 0.0)))
        out: Dict[str, int] = {}
        for o in run(g, int(24 * 3600 / dt_s), dt_s, aggregated=False):
            if o.app_proto == "dns":
                out[o.entity] = out.get(o.entity, 0) + 1
        return out
    d60, d900 = dns(60.0), dns(900.0)
    assert set(d60) == set(d900) == {"10.40.4.51", "10.20.9.5"}
    for k in d900:
        assert abs(d60[k] - d900[k]) <= 2, (k, d60[k], d900[k])
    assert d900["10.20.9.5"] <= 2                        # one lookup per backup run


def test_human_sessions_carry_across_tick_boundaries():
    g = TrafficGenerator(seed=1, pack=MiniPack(population=HUMANS[:2],
                                               start_epoch=local_epoch(MONDAY, 9.0)))
    for _ in range(30):
        t0 = g.vt
        for o in g.step(60.0, aggregated=False):
            assert t0 <= o.ts < g.vt                   # carried events land in their own tick
    assert any(k.startswith("hs|") and v for k, v in g._state.items())


def test_aggregated_retransmits_are_a_total_that_r1_does_not_reweight():
    """An aggregate's per-record fields are per-flow values (R1 adds w x
    field); the group's retransmit total travels in extra['retransmits_total'].
    It used to sit in the per-flow field, so R1 counted it w times and
    l4.retransmit_rate of aggregated ticks was ~w x the event-mode rate
    (pack E: warm-up aggregated, live in event mode)."""
    from helpers import make_store, run_engine
    from app.engines.raw.l4flow import L4FlowEngine
    pk = get_pack("mini")
    ga = TrafficGenerator(seed=3, pack=pk)
    gr = TrafficGenerator(seed=3, pack=get_pack("mini"))
    ga.vt = gr.vt = local_epoch(MONDAY, 8.0)
    sa, sr = make_store(), make_store()
    rates = {"agg": [0.0, 0.0], "raw": [0.0, 0.0]}
    for _ in range(16):
        agg = ga.step(900.0)
        raw = gr.step(900.0, aggregated=False)
        for o in agg:
            if o.app_proto == "http":
                assert o.extra["retransmits_total"] >= 0
                assert o.retransmits == round(o.extra["retransmits_total"] / o.extra["count"])
            else:
                assert "retransmits_total" not in o.extra
        for tag, st, obs in (("agg", sa, agg), ("raw", sr, raw)):
            run_engine(L4FlowEngine(), st, ga.vt, observations=obs)
            for s in st.systems():
                for e in st.entities(s):
                    m = st.raw_tail(s, e, "l4.retransmit_rate", 1)
                    n = st.raw_tail(s, e, "l4.pkts_total", 1)
                    if m and n and m[-1].ts == ga.vt:
                        rates[tag][0] += m[-1].value * n[-1].value
                        rates[tag][1] += n[-1].value
    ra = rates["agg"][0] / rates["agg"][1]
    rr = rates["raw"][0] / rates["raw"][1]
    assert rr > 0 and 0.7 < ra / rr < 1.4


def test_aggregated_counts_match_fine_cadence_in_expectation():
    keys = HUMANS + ["erp-prod|10.20.4.30", "api-gateway|10.40.4.53"]

    def total(dt_s, seed):
        g = TrafficGenerator(seed=seed, pack=MiniPack(population=keys,
                                                      start_epoch=local_epoch(MONDAY, 7.0)))
        return sum(w(o) for o in run(g, int(12 * 3600 / dt_s), dt_s) if o.app_proto == "http")
    coarse = np.mean([total(900.0, s) for s in range(3)])
    fine = np.mean([total(60.0, s) for s in range(3)])
    hourly = np.mean([total(3600.0, s) for s in range(3)])
    assert abs(coarse / fine - 1) < 0.1 and abs(hourly / fine - 1) < 0.1


# --------------------------------------------------------------------------- #
# v1 compatibility (build.Runtime)
# --------------------------------------------------------------------------- #
def test_backward_compatible_step_and_demo():
    g = TrafficGenerator(window_s=60)
    assert set(g.personas) == set(SMOKE_KEYS) and len(SMOKE_KEYS) == 20
    g.vt = local_epoch(MONDAY, 9.0)
    obs = g.step(dt=900, live=False)
    assert obs and all(isinstance(o, G.Observation) for o in obs)
    assert g.truth == []
    for _ in range(12):
        g.step(dt=60, live=True)
    ids = {r["scenario_id"] for r in g.truth}
    assert {"demo-exfil", "T1-demo", "T12-demo"} <= ids
    assert any(o.entity == "10.30.2.99" for o in g.step(dt=60, live=True))


def test_hr_salary_export_is_used_by_hr_only():
    keys = ["erp-prod|10.20.1.21", "erp-prod|10.20.1.13", "erp-prod|10.20.1.11"]
    g = TrafficGenerator(seed=0, pack=MiniPack(population=keys,
                                               start_epoch=local_epoch(MONDAY, 0.0)))
    obs = run(g, 5 * 96, 900.0)
    by = {k: sum(w(o) for o in obs if f"{o.system}|{o.entity}" == k
                 and (o.http_path or "").startswith("/hr/salary/export")) for k in keys}
    assert by["erp-prod|10.20.1.21"] >= 20                 # df = 1 of the class (T7)
    assert by["erp-prod|10.20.1.13"] == by["erp-prod|10.20.1.11"] == 0


def test_human_uploads_are_heavy_tailed():
    g = TrafficGenerator(seed=0, pack=MiniPack(population=["erp-prod|10.20.1.12"],
                                               start_epoch=local_epoch(MONDAY, 0.0)))
    per = []
    for _ in range(5 * 96):
        o = g.step(900.0)
        if any(x.app_proto == "http" for x in o):
            per.append(math.log1p(sum(x.extra["bytes_up_total"] for x in o)))
    per = np.array(per)
    assert len(per) > 100 and 0.6 < per.std() < 1.6        # wide per-tick marginal


def test_browser_update_bumps_the_browser_minor_version_only():
    ua = G.ORG_STANDARD.ua
    up = G.bump_minor(ua)
    assert up.startswith("Mozilla/5.0 (Windows NT 10.0;") and "Chrome/126.1.0.0" in up
    ff = G.UA_FF_LIN.format(v=128)
    assert G.bump_minor(ff).endswith("rv:128.1) Gecko/20100101 Firefox/128.1")
    sf = G.UA_SAFARI.format(v=17)
    assert "Version/17.6 Safari" in G.bump_minor(sf) and "Mac OS X 14_5" in G.bump_minor(sf)
