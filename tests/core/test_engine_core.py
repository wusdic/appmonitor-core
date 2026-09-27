"""Engine core v2 (contract I/M, architecture §10): config defaults, period_s
scheduling, per-entity phased refits, safe_run health + strict re-raise, and
the orchestrator's dt / config / ops.engine_health plumbing."""
import pytest

from helpers import DT, T0, ctx, make_store, run_engine

from app.core.engine import DEFAULT_CONFIG, Context, Engine, Registry
from app.models.schema import RawMetric
from app.pipeline.orchestrator import Pipeline

S, E = "sys", "10.0.0.1"


class Counter(Engine):
    name = "test.counter"
    layer = "behavior"

    def __init__(self, **p):
        super().__init__(**p)
        self.seen = []

    def run(self, ctx, observations=None):
        self.seen.append((ctx.now, ctx.window_s, ctx.training))
        return 1


class Boom(Engine):
    name = "test.boom"
    layer = "behavior"

    def run(self, ctx, observations=None):
        raise RuntimeError("kaput")


class Writer(Engine):
    name = "test.writer"
    layer = "raw"

    def run(self, ctx, observations=None):
        for o in observations or []:
            ctx.store.add_raw(RawMetric("l4.flows", 1, ctx.now, o.system, o.entity))
        return len(observations or [])


# ------------------------------------------------------------------ context
def test_context_config_defaults_and_overrides():
    c = Context(store=make_store(), now=T0)
    assert c.config["tz"] == "Asia/Shanghai"
    assert c.config["D_min_s"] == 600
    assert c.config["daypart_day_hours"] == [8, 20]
    assert c.config["strict"] is False and c.strict is False
    assert c.config["alert_budget"] == {"entity_per_hour": 3, "system_per_day": 20}
    assert c.config["calendar"] == {"holidays": [], "makeup_workdays": []}
    user = {"strict": True, "tz": "UTC"}
    c2 = Context(store=make_store(), now=T0, window_s=900, training=True, config=user)
    assert c2.config["tz"] == "UTC" and c2.strict and c2.config["D_min_s"] == 600
    assert c2.training and c2.dt == 900
    assert user == {"strict": True, "tz": "UTC"}                  # caller dict untouched
    c.config["calendar"]["holidays"].append("x")                  # defaults are deep-copied
    assert DEFAULT_CONFIG["calendar"]["holidays"] == []


# ------------------------------------------------------------------ scheduling
def _drive(engine, times, dt=DT):
    st = make_store()
    ran = []
    for t in times:
        before = len(engine.seen)
        engine.safe_run(ctx(st, t, window_s=dt))
        if len(engine.seen) > before:
            ran.append(t)
    return ran


def test_interval_semantics_unchanged():
    e = Counter(interval=4)
    ran = _drive(e, [T0 + i * DT for i in range(12)])
    assert ran == [T0, T0 + 4 * DT, T0 + 8 * DT]                  # calls 1, 5, 9 as in v1


def test_period_s_triggers_before_interval():
    # interval 16 ticks, period 1 h: at 900 s ticks the period (4 ticks) wins
    e = Counter(interval=16, period_s=3600)
    ran = _drive(e, [T0 + i * 900 for i in range(13)])
    assert ran == [T0, T0 + 3600, T0 + 7200, T0 + 10800]
    # at 60 s ticks the interval (16 min) wins over the 1 h period
    e2 = Counter(interval=16, period_s=3600)
    ran2 = _drive(e2, [T0 + i * 60 for i in range(40)], dt=60)
    assert ran2 == [T0, T0 + 16 * 60, T0 + 32 * 60]


def test_period_s_class_attribute_and_clock_reset():
    class Slow(Counter):
        interval = 1000
        period_s = 7200.0
    e = Slow()
    assert e.period_s == 7200.0
    ran = _drive(e, [T0, T0 + 3600, T0 + 7200, T0 + 3600 * 2.5])
    assert ran == [T0, T0 + 7200]
    # a clock that goes backwards (new run / replay) makes the engine due
    assert e.is_due(T0 - 10)


def test_entity_due_phased_and_deterministic():
    e = Counter(period_s=6 * 3600)
    keys = [f"sys|10.0.0.{i}" for i in range(40)]
    first = {k: e.entity_due(k, T0) for k in keys}
    assert all(first.values())                                     # first call always due
    fire_times = {k: [] for k in keys}
    for i in range(1, 49):                                         # 12 h at 900 s
        t = T0 + i * 900
        for k in keys:
            if e.entity_due(k, t):
                fire_times[k].append(t)
    # each key refits once per 6 h window: 2 times in 12 h
    assert all(len(v) == 2 for v in fire_times.values())
    # ... spaced by exactly the period
    assert all(v[1] - v[0] == 6 * 3600 for v in fire_times.values())
    # ... and spread over the period rather than all on one tick
    assert len({v[0] for v in fire_times.values()}) > 10
    # deterministic across instances
    e2 = Counter(period_s=6 * 3600)
    for k in keys:
        e2.entity_due(k, T0)
    assert [e2.entity_due(k, fire_times[k][0]) for k in keys] == [True] * len(keys)
    # no period -> always due; explicit period overrides
    assert Counter().entity_due("x", T0) and Counter().entity_due("x", T0)
    assert e.entity_due(("sys", "e"), T0, period_s=60) and not e.entity_due(("sys", "e"), T0 + 1, period_s=60)


# ------------------------------------------------------------------ safe_run / health
def test_safe_run_records_health_nonstrict():
    st = make_store()
    b = Boom()
    assert b.safe_run(ctx(st, T0)) == 0
    assert b.safe_run(ctx(st, T0 + DT)) == 0
    assert b.error_count == 2 and b.last_error_ts == T0 + DT
    assert "RuntimeError: kaput" in b.last_error
    assert "kaput" in b.last_traceback and "test_engine_core.py" in b.last_traceback
    h = st.health()["test.boom"]
    assert h["ok"] is False and h["error_count"] == 2 and h["last_error_ts"] == T0 + DT
    assert h["ts"] == T0 + DT and "RuntimeError" in h["traceback"]
    assert st.engine_failed("test.boom", T0 + DT) and not st.engine_failed("test.boom", T0)
    assert not st.engine_failed("nosuch", T0)
    c = Counter()
    c.safe_run(ctx(st, T0))
    hc = st.health()["test.counter"]
    assert hc["ok"] is True and hc["error_count"] == 0 and hc["last_count"] == 1
    assert hc["duration_ms"] >= 0
    info = b.info()
    assert info["error_count"] == 2 and info["last_error_ts"] == T0 + DT


def test_safe_run_strict_reraises():
    st = make_store()
    b = Boom()
    with pytest.raises(RuntimeError):
        b.safe_run(ctx(st, T0, config={"strict": True}))
    assert b.error_count == 1 and st.health()["test.boom"]["ok"] is False
    with pytest.raises(RuntimeError):
        run_engine(Boom(), st, T0)                                 # helper is strict
    assert run_engine(Counter(), st, T0) == 1


def test_safe_run_tolerates_store_without_health():
    class Bare:
        pass
    c = Counter()
    assert c.safe_run(Context(store=Bare(), now=T0)) == 1


def test_run_engine_helper_passes_dt_training_config():
    st = make_store()
    c = Counter(interval=5)
    run_engine(c, st, T0, training=True, dt=60, config={"tz": "UTC"})
    run_engine(c, st, T0 + 60, dt=60)                              # bypasses interval
    assert c.seen == [(T0, 60, True), (T0 + 60, 60, False)]
    assert run_engine(c, st, T0 + 120, scheduled=True) == 1       # through safe_run:
    assert run_engine(c, st, T0 + 180, scheduled=True) == 0       # interval applies


# ------------------------------------------------------------------ orchestrator
def test_pipeline_dt_config_and_engine_health():
    from helpers import obs
    st = make_store()
    reg = Registry()
    cnt, boom = Counter(), Boom()
    reg.add(Writer(), cnt, boom)
    p = Pipeline(st, reg, window_s=60, config={"tz": "UTC"})
    assert p.config["tz"] == "UTC" and p.config["D_min_s"] == 600
    stats = p.run_tick([obs(S, E, T0 - 5)], now=T0, training=True, dt=900)
    assert stats == {"test.writer": 1, "test.counter": 1, "test.boom": 0}
    assert cnt.seen[-1] == (T0, 900, True)
    p.run_tick([], now=T0 + 60)                                    # default dt = window_s
    assert cnt.seen[-1] == (T0 + 60, 60, False)
    assert st.last_seen(S, E) == T0
    eh = st.latest_derived(S, "__system__", "ops.engine_health")
    assert eh.ts == T0 + 60 and eh.value["n_errors"] == 1 and "test.boom" in eh.value["errors"]
    assert st.entities(S) == [E] and "__system__" in st.pseudo_entities(S)
    # strict config propagates and re-raises out of the tick
    p2 = Pipeline(make_store(), reg, config={"strict": True})
    with pytest.raises(RuntimeError):
        p2.run_tick([], now=T0)
