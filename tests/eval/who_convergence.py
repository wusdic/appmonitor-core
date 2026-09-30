"""Who-discovery on the generator's own organisation (pack O, all systems;
docs/lib3/progressive.md §6.15, requirement S5/S11 "综合部的3个用户", S3
"用的时间越长越精准").

Traffic: orggen pack O through R2, R3, P00 (the real raw engines) and P11
(behaviour groups from the IP x action co-occurrence of the learned rows;
without P03 / P04 the items are action routes only). At each checkpoint day
the learned groups are scored against the departments the generator
published (group truth), with nothing read by an engine:
  ari        adjusted Rand index over the static departments' IPs (GA, FIN,
             SALES), singletons for ungrouped IPs (pmetrics.ari, PG3)
  pool       share of the DEV DHCP pool's IPs seen in the last 7 days that sit
             in one group or one prefix cover (PG3 'dev_pool_grouped')
  named      the configured name 综合部 attached to a group whose members are
             exactly 综合部's IPs
  cost       P11 state bytes and ms per tick."""
from __future__ import annotations

import ipaddress
import time
from typing import Any, Dict, List, Optional, Sequence

from app.core.engine import Context
from app.core.store import MetricStore
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.who_groups import STATE, WhoGroupsEngine
from app.engines.raw.action_token import ActionTokenEngine
from app.engines.raw.client_stack import ClientStackEngine
from app.engines.raw.event_builder import EventBuilderEngine
from app.eval.pmetrics import ari
from app.models.schema import ORG
from app.pipeline.orggen import OrgGenerator, build_org


def measure(store: Any, gen: OrgGenerator, now: float) -> Dict[str, Any]:
    wg = MP.who_groups(store)
    ip2g = wg.get("ip2g") or {}
    t, p = [], []
    for d in gen.spec.departments:
        if d.kind != "static":
            continue
        for ip in d.ips:
            t.append(d.code)
            p.append(ip2g.get(ip, f"_:{ip}"))
    st = store.get_model(ORG, ORG, STATE)
    dev = next(d for d in gen.spec.departments if d.code == "DEV")
    pool = ipaddress.ip_network(dev.pool[0], strict=False)
    seen = [k for k in (st.sigs.keys() if st is not None else [])
            if "/" not in str(k) and ipaddress.ip_address(k) in pool]
    labs = [ip2g.get(ip) for ip in seen]
    one = max((labs.count(x) for x in set(labs) if x is not None), default=0) / len(seen) if seen else None
    covers = [ipaddress.ip_network(c) for cs in (wg.get("covers") or {}).values() for c in cs]
    one_prefix = any(n.version == pool.version and (n == pool or (pool.subnet_of(n) and
                     n.prefixlen >= pool.prefixlen - 2)) for n in covers)
    ga = next(d for d in gen.spec.departments if d.code == "GA")
    named = any(r.get("name") == ga.name and set(r.get("members") or []) == set(ga.ips)
                for r in (wg.get("groups") or {}).values())
    return {"ari": float(ari(t, p)) if len(t) > 1 else None, "pool_one_group": one,
            "pool_grouped": bool((one or 0) >= 0.9 or one_prefix), "named_ga": named,
            "groups": len(wg.get("groups") or {}), "state_kb": round(st.nbytes() / 1024.0, 1) if st else 0}


def run(days: Sequence[int] = (2, 4, 8), dt: float = 3600.0, seed: int = 0,
        keep: Optional[Dict[str, Any]] = None) -> Dict[int, Dict[str, Any]]:
    spec = build_org("O")
    gen = OrgGenerator(spec, seed=seed, pack_name="O")
    cfg = {"grain_mode": "canonical", "strict": True, "tz": spec.tz, "calendar": dict(spec.calendar)}
    cfg.update(spec.config)
    st = MetricStore()
    p11 = WhoGroupsEngine()
    engines = [ActionTokenEngine(), ClientStackEngine(), EventBuilderEngine(), p11]
    t0 = gen.day_start(1)
    per_day = int(round(86400.0 / dt))
    out: Dict[int, Dict[str, Any]] = {}
    ms = 0.0
    for k in range(max(days) * per_day):
        a, b = t0 + k * dt, t0 + (k + 1) * dt
        obs = gen.step(a, b, aggregated=dt >= 900)
        c = Context(store=st, now=b, window_s=dt, config=cfg)
        for e in engines:
            x = time.perf_counter()
            e.safe_run(c, obs if e.layer == "raw" else None)
            if e is p11:
                ms += (time.perf_counter() - x) * 1000.0
        if (k + 1) % per_day == 0 and (k + 1) // per_day in days:
            d = (k + 1) // per_day
            out[d] = dict(measure(st, gen, b), p11_ms_per_tick=round(ms / (k + 1), 2))
    if keep is not None:
        keep.update(store=st, gen=gen)
    return out


if __name__ == "__main__":                                  # pragma: no cover
    import sys
    for s in (int(x) for x in (sys.argv[1:] or ["0"])):
        print(s, run(days=(1, 2, 3, 5, 8, 14), seed=s))
