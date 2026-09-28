"""Part B of the cadence-invariance property test (docs/lib3/cadence.md
§13.2; slow, run by tests/test_cadence_invariance.py only with APPMON_SLOW=1).

Six control personas (4 human, 2 machine, no scenarios) warm up by the §10
rule for a Monday start (96 x 3600 s + 288 x 900 s, Fri..Sun), then run 12
live hours (09:00-21:00) at dt = 60 s. The twin run replays the SAME live
observations (the generator is stepped at 60 s in both runs, so its random
stream is identical) re-batched into 900-s ticks. Both runs are canonical.
"""
from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Dict, List, Tuple

import numpy as np

from app.core.engine import default_config
from app.core.store import MetricStore
from app.engines.behavior.lib import features as F
from app.eval import packs as P
from app.pipeline.build import build_registry, load_signatures
from app.pipeline.generator import TrafficGenerator
from app.pipeline.orchestrator import Pipeline

KEYS = ["erp-prod|10.20.1.11", "erp-prod|10.20.1.12", "erp-prod|10.20.1.13",
        "erp-prod|10.20.1.15", "erp-prod|10.20.9.9", "api-gateway|10.40.4.51"]
CAL = {"holidays": [], "makeup_workdays": []}
LIVE_S = 12 * 3600.0
_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def _sev(x: Any) -> int:
    return _RANK.get(str(getattr(x, "value", x)).lower(), 0)


def _pack() -> P.Pack:
    phases = P._phases((96, 3600.0), (288, 900.0), (int(LIVE_S // 60), 60.0))
    tl = P.Timeline(P.SHANGHAI, CAL, _dt.date(2025, 3, 10), phases, scen_hour=9.0)
    return P._pack("partB", tl, KEYS, [], "cadence part B", backgrounds=False)


def _run(live_dt: float, seed: int = 0, mode: str = "canonical") -> Dict[str, Any]:
    pk = _pack()
    cfg = default_config({"tz": P.SHANGHAI, "calendar": CAL, "strict": True,
                          "grain_mode": mode})
    sig, comp = load_signatures()
    pl = Pipeline(MetricStore(), build_registry(sig, comp, config=cfg), window_s=60, config=cfg)
    gen = TrafficGenerator(seed=seed, pack=pk)
    for n, dt, _agg in pk.phases[:-1]:
        for _ in range(n):
            # non-aggregated like the live phase: aggregated generator records
            # (one record per (key, template) and tick) change the stream-
            # timestamp and session features the live events are compared to
            obs = gen.step(dt, aggregated=False)
            pl.run_tick(obs, now=gen.vt, training=True, dt=dt)
    t_live = gen.vt
    buf: List[Any] = []
    k = int(round(live_dt / 60.0))
    for i in range(int(LIVE_S // 60)):
        buf.extend(gen.step(60.0, live=True, aggregated=False))
        if (i + 1) % k == 0:
            pl.run_tick(buf, now=gen.vt, training=False, dt=live_dt)
            buf = []
    return {"store": pl.store, "t_live": t_live, "t_end": gen.vt}


def _stats(r: Dict[str, Any]) -> Dict[str, Any]:
    st, t0 = r["store"], r["t_live"]
    days = LIVE_S / 86400.0
    single = 0
    acc_b14 = 0
    ents = [(s, e) for s in st.systems() for e in st.entities(s)]
    for s, e in ents:
        ts, E = st.vec_since(s, e, "behavior.e_day", t0 + 1e-6)
        single += int(np.sum(np.asarray(E, dtype=float).reshape(-1) <= 0.03))
        for m in st.derived_tail(s, e, "behavior.acc_alarm", 10 ** 6):
            if m.ts > t0 and isinstance(m.value, dict) and any(
                    m.value.get(d) for d in ("cusum", "mcusum")):
                acc_b14 += 1
    incs = [i for i in st.incidents() if float(i.opened) > t0]
    ident = [ev for ev in st.events(since=t0 + 1e-6, limit=10 ** 9)
             if ev.kind in ("unknown_identity", "identity_mismatch") and _sev(ev.severity) >= 2]
    return {"single_tick_alarms": single,
            "incidents_medium_plus": sum(1 for i in incs if _sev(i.severity) >= 2),
            "identity_medium_plus": len(ident), "acc_alarms_b14": acc_b14,
            "far_low": sum(1 for i in incs if _sev(i.severity) >= 1) / max(1e-9,
                                                                           len(ents) * days)}


def _h_mismatch(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    """Common H rows whose add-class features differ beyond 1e-5 relative."""
    bad = 0
    sa, sb = a["store"], b["store"]
    add = [i for i, n in enumerate(F.FEATURE_NAMES_V2) if F.GRAIN_CLASS.get(n) == "add"]
    for s in sa.systems():
        for e in sa.entities(s):
            ta, A = sa.vec_since(s, e, "feature.nat.h", a["t_live"] + 1e-6)
            tb, B = sb.vec_since(s, e, "feature.nat.h", b["t_live"] + 1e-6)
            rb = {float(t): np.asarray(r, dtype=float) for t, r in zip(tb, B)}
            for t, ra in zip(ta.tolist(), A):
                x = rb.get(float(t))
                if x is None:
                    bad += 1
                    continue
                ra = np.asarray(ra, dtype=float)
                for i in add:
                    u, v = ra[i], x[i]
                    if math.isnan(u) and math.isnan(v):
                        continue
                    if not abs(u - v) <= 1e-5 * max(1.0, abs(u), abs(v)):
                        bad += 1
                        break
    return bad


def run_part_b(seed: int = 0) -> Dict[Any, Any]:
    r60 = _run(60.0, seed)
    r900 = _run(900.0, seed)
    out: Dict[Any, Any] = {60.0: _stats(r60), 900.0: _stats(r900)}
    out["h_row_mismatch"] = _h_mismatch(r60, r900)
    return out


if __name__ == "__main__":        # pragma: no cover
    import json
    print(json.dumps({str(k): v for k, v in run_part_b().items()}, indent=1))
