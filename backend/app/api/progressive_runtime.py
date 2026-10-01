"""A Runtime that runs the progressive profile core (docs/lib3/progressive.md
§9.1, §11) behind the API, for the "画像模式" pages (routes_v3.py).

The default Runtime (main.py) runs the full B-library on the v2 generator,
whose records carry no per-event payload view, so the P engines would have
nothing to learn from. This factory builds the same Runtime class on an
organisation pack (pack O by default: 综合部 / 财务部 / 销售部 / 研发 / 门户 on
OA, finance, CRM, portal, mail and code systems), registers the requested
engine set, and warms up `days` days at 900-s aggregated ticks with
`ev_sample` (the pack's own phase), after which the live loop continues at the
runtime's window (60-s event-mode ticks of the same organisation).

Registry modes (build.registry_mode): 'progressive_decision' (default: the
P-core plus B24-B29, ~35 s of warm-up per simulated day and ~1.4 GB on 21
days of pack O, progressive.md §16.3), 'progressive_only' (P-core alone) and
'full+progressive' (everything; > 2 GB by day 4 on pack O, not for a demo).

Environment (main.py): APPMON_PROGRESSIVE=decision|only|full (unset = the
classic runtime), APPMON_PACK (default 'O'), APPMON_PROGRESSIVE_DAYS
(default 7), APPMON_SEED (default 0).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ..pipeline.build import Runtime, build_registry, registry_mode
from ..pipeline.orchestrator import Pipeline

MODES = {"decision": "progressive_decision", "only": "progressive_only",
         "full": "full+progressive", "1": "progressive_decision", "true": "progressive_decision",
         "progressive_decision": "progressive_decision", "progressive_only": "progressive_only",
         "full+progressive": "full+progressive"}
DEFAULT_DAYS = 7
TICKS_PER_DAY = 96          # 900-s aggregated ticks (the org packs' own phase)


def resolve_mode(value: Optional[str]) -> Optional[str]:
    """APPMON_PROGRESSIVE value -> registry mode (None = progressive off)."""
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ("", "0", "false", "off", "no"):
        return None
    if v not in MODES:
        raise ValueError(f"APPMON_PROGRESSIVE={value!r}; expected one of {sorted(set(MODES))}")
    return MODES[v]


def make_progressive_runtime(mode: str = "progressive_decision", pack_name: str = "O",
                             days: float = DEFAULT_DAYS, seed: int = 0,
                             live_period_s: float = 3.0, window_s: int = 60,
                             config: Optional[Dict[str, Any]] = None) -> Runtime:
    """A (not yet warmed) Runtime on an organisation pack with `mode`'s engines."""
    from ..eval.packs import get_pack
    mode = MODES.get(mode, mode)
    pack = get_pack(pack_name)
    if getattr(pack, "org", None) is None:
        raise ValueError(f"pack {pack_name!r} is not an organisation pack (progressive.md §11.6)")
    cfg: Dict[str, Any] = dict(pack.config or {})
    cfg.setdefault("progressive", {})
    cfg["progressive"] = dict(cfg["progressive"], enabled=True)
    cfg.update(config or {})
    n = max(1, int(round(float(days) * TICKS_PER_DAY)))
    rt = Runtime(window_s=window_s, live_period_s=live_period_s, config=cfg, seed=seed, pack=pack,
                 warmup_plan=[(n, 900.0)])
    if registry_mode(mode) != registry_mode(None, rt.config):
        rt.registry = build_registry(rt.sig_store, rt.composite_rules, config=rt.config,
                                     pack=pack, progressive=mode)
        rt.pipeline = Pipeline(rt.store, rt.registry, window_s=window_s, config=rt.config)
    rt.registry_mode = registry_mode(mode)          # read by GET /api/v3/status
    rt.pack_name = str(pack_name)
    return rt


def runtime_from_env(env: Optional[Dict[str, str]] = None) -> Optional[Runtime]:
    """The progressive Runtime main.py starts when APPMON_PROGRESSIVE is set."""
    env = dict(os.environ if env is None else env)
    mode = resolve_mode(env.get("APPMON_PROGRESSIVE"))
    if mode is None:
        return None
    return make_progressive_runtime(
        mode, pack_name=env.get("APPMON_PACK", "O"),
        days=float(env.get("APPMON_PROGRESSIVE_DAYS", DEFAULT_DAYS)),
        seed=int(env.get("APPMON_SEED", "0")),
        live_period_s=float(env.get("APPMON_LIVE_PERIOD_S", "3.0")))
