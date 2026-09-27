"""Assembly: build the store, the full engine registry, and the runtime.

This is the one place that knows the concrete engine set. Everything else
depends only on the Pipeline / Registry / Store abstractions.

The registry follows docs/lib3/architecture.md §1 exactly: layers run raw ->
derived -> behavior -> signature, and within a layer engines run in the order
they are registered here. Dependencies that the order below satisfies inside
the same tick (producer before consumer):

  raw        R1 (l2l3 .. probe) -> R2 action_token -> R3 client_stack
  derived    D0 (aggregation, periodicity, trend) -> D1 (ratio, entropy, graph)
             -> D2 session
  behavior   B01 feature_vector (the activity clock every learner gates on)
             -> B02 peer_group -> B03 baseline -> B04 likelihood (behavior.z)
             -> B05 common_mode (reads z) -> B06 .. B18 detectors
             -> B23 feedback -> B24 calibration (behavior.p) -> B25 fusion
             -> B26 risk -> B27 incident -> B28 governor -> [B29 explain]
             -> B30 portrait
  signature  rule_match -> correlation

One-tick lags the spec allows (and the engines are written for):
  * B05 reads B08 novelty events of t-1 (B08 runs after it);
  * every learner reads trust / quarantine of t-1 from B28 (commit row t-D);
  * B26 / B27 / B28 read signature matches with a one-tick lag.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

import yaml

from ..core.engine import Registry, default_config
from ..core.store import MetricStore
# behavior (lib-3, v2)
from ..engines.behavior.attribution import AttributionEngine
from ..engines.behavior.baseline import BaselineEngine
from ..engines.behavior.beacon import BeaconEngine
from ..engines.behavior.budget import BudgetEngine
from ..engines.behavior.calibration import CalibrationEngine
from ..engines.behavior.changepoint import ChangepointEngine
from ..engines.behavior.class_monitor import ClassMonitorEngine
from ..engines.behavior.client_identity import ClientIdentityEngine
from ..engines.behavior.common_mode import CommonModeEngine
from ..engines.behavior.entity_link import EntityLinkEngine
from ..engines.behavior.feature_vector import FeatureVectorEngine
from ..engines.behavior.feedback import FeedbackEngine
from ..engines.behavior.fusion import FusionEngine
from ..engines.behavior.governor import GovernorEngine
from ..engines.behavior.identity_model import IdentityModelEngine
from ..engines.behavior.incident import IncidentEngine
from ..engines.behavior.likelihood import LikelihoodEngine
from ..engines.behavior.multivariate import MultivariateEngine
from ..engines.behavior.novelty import NoveltyEngine
from ..engines.behavior.peer_group import PeerGroupEngine
from ..engines.behavior.portrait import PortraitEngine
from ..engines.behavior.rhythm import RhythmEngine
from ..engines.behavior.risk import RiskEngine
from ..engines.behavior.sequence import SequenceEngine
from ..engines.behavior.timing import TimingEngine
# derived (lib-2)
from ..engines.derived.aggregation import AggregationEngine
from ..engines.derived.entropy import EntropyEngine
from ..engines.derived.graph import GraphEngine
from ..engines.derived.periodicity import PeriodicityEngine
from ..engines.derived.ratio import RatioEngine
from ..engines.derived.session import SessionEngine
from ..engines.derived.trend import TrendEngine
# raw (lib-1)
from ..engines.raw.action_token import ActionTokenEngine
from ..engines.raw.active_probe import ActiveProbeEngine
from ..engines.raw.client_stack import ClientStackEngine
from ..engines.raw.dns import DNSEngine
from ..engines.raw.http import HTTPEngine
from ..engines.raw.l2l3 import L2L3Engine
from ..engines.raw.l4flow import L4FlowEngine
from ..engines.raw.tls import TLSEngine
# signature (lib-4)
from ..engines.signature.correlation import CorrelationEngine
from ..engines.signature.rule_match import RuleMatchEngine
from ..engines.signature.store import SignatureStore
from .generator import TrafficGenerator
from .orchestrator import Pipeline

DATA_DIR = os.environ.get(
    "APPMON_SIGNATURE_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "signatures"))

# Slot reserved for B29 explain (docs/lib3/engines.md B29): it runs right after
# B28 governor. When the engine lands, add it to `_explain_engines()`.
EXPLAIN_SLOT_AFTER = "behavior.governor"


def limit_blas_threads(n: int = 1) -> None:
    """Pin BLAS/OpenMP to `n` threads (integration notes R15.1 / R19.3).

    The lib-3 fits are many tiny (<= 200 x 200) eigendecompositions and
    C-steps; multithreaded OpenBLAS makes them 3-10x slower under load. The
    env vars cover libraries loaded later; threadpoolctl covers the ones that
    are already loaded (numpy's OpenBLAS is, by the time this runs)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n)
    except Exception:  # pragma: no cover - threadpoolctl ships with sklearn
        pass


def _explain_engines() -> list:
    """B29 explain (P1, not yet implemented): the registry slot after B28."""
    return []


def build_registry(sig_store: Optional[SignatureStore] = None, composite_rules=None, *,
                   config: Optional[Dict[str, Any]] = None, pack: Any = None,
                   seed: int = 0) -> Registry:
    """The production registry, in architecture.md §1 order.

    `sig_store` / `composite_rules` default to the files under DATA_DIR.
    `config`, `pack` and `seed` are accepted for the eval runner's factory
    seam (eval/runner.default_registry_factory); engines read the runtime
    config from ctx.config, so nothing here depends on them."""
    if sig_store is None or composite_rules is None:
        s2, c2 = load_signatures()
        sig_store = s2 if sig_store is None else sig_store
        composite_rules = c2 if composite_rules is None else composite_rules
    limit_blas_threads(1)
    reg = Registry()
    # raw (原始指标库): R1 full sets, then R2 action tokens and R3 client stacks
    reg.add(L2L3Engine(), L4FlowEngine(), HTTPEngine(), TLSEngine(), DNSEngine(),
            ActiveProbeEngine(),
            ActionTokenEngine(),            # R2
            ClientStackEngine())            # R3
    # derived (次生指标库): D0 window, D1 instant, D2 session
    reg.add(AggregationEngine(), PeriodicityEngine(), TrendEngine(),   # D0
            RatioEngine(), EntropyEngine(), GraphEngine(),             # D1
            SessionEngine())                                           # D2
    # behaviour (行为库), B01 .. B30 in contract order
    reg.add(FeatureVectorEngine(),          # B01
            PeerGroupEngine(),              # B02 (own 16-tick / 6 h refit stride)
            BaselineEngine(),               # B03 (interval 1: commits row t-D)
            LikelihoodEngine(),             # B04
            CommonModeEngine(),             # B05
            MultivariateEngine(),           # B06
            RhythmEngine(),                 # B07
            NoveltyEngine(),                # B08
            ClientIdentityEngine(),         # B09
            SequenceEngine(),               # B10
            TimingEngine(),                 # B11
            BeaconEngine(),                 # B12
            BudgetEngine(),                 # B13
            ChangepointEngine(),            # B14
            IdentityModelEngine(),          # B15 (own 96-tick / 24 h fit stride)
            AttributionEngine(),            # B16
            EntityLinkEngine(),             # B17
            ClassMonitorEngine(),           # B18
            # P2 B19-B22 are gated by the ablation gate and not registered
            FeedbackEngine(),               # B23
            CalibrationEngine(),            # B24
            FusionEngine(),                 # B25
            RiskEngine(),                   # B26
            IncidentEngine(),               # B27
            GovernorEngine(),               # B28
            *_explain_engines(),            # B29 slot
            PortraitEngine())               # B30 (every tick; each key refreshes per 2 h at its own phase)
    # signature (行为特征库)
    reg.add(RuleMatchEngine(sig_store, min_confidence=0.6),
            CorrelationEngine(composite_rules))
    return reg


def load_signatures():
    sig_store = SignatureStore()
    prim = os.path.join(DATA_DIR, "primitives.yaml")
    if os.path.exists(prim):
        sig_store.load_file(prim)
    composite_rules = []
    comp = os.path.join(DATA_DIR, "composite.yaml")
    if os.path.exists(comp):
        with open(comp, "r", encoding="utf-8") as f:
            composite_rules = yaml.safe_load(f) or []
    return sig_store, composite_rules


class Runtime:
    """Owns the pipeline + generator, warms up history, and runs a live loop
    in a background thread. The API reads the store this exposes.

    Contract I: the pipeline config carries tz, calendar and strict; warm-up
    runs with training=True at Δt = 900 s, the live loop at the runtime's
    window, and every tick is stamped `now = gen.vt` (architecture §10)."""

    WARMUP_DT = 900.0

    def __init__(self, window_s: int = 60, warmup_ticks: int = 180, live_period_s: float = 3.0,
                 config: Optional[Dict[str, Any]] = None, seed: int = 42,
                 strict: bool = False, pack: Any = None):
        self.store = MetricStore()
        self.sig_store, self.composite_rules = load_signatures()
        self.gen = TrafficGenerator(seed=seed, window_s=window_s, pack=pack)
        clock = self.gen.clock
        cfg: Dict[str, Any] = {"tz": clock.tz, "calendar": {
            "holidays": sorted(d.isoformat() for d in clock.holidays),
            "makeup_workdays": sorted(d.isoformat() for d in clock.makeup)}}
        cfg.update(config or {})
        cfg["strict"] = bool(cfg.get("strict", strict))
        self.config = default_config(cfg)
        self.registry = build_registry(self.sig_store, self.composite_rules, config=self.config)
        self.pipeline = Pipeline(self.store, self.registry, window_s=window_s,
                                 config=self.config)
        self.window_s = window_s
        self.warmup_ticks = warmup_ticks
        self.live_period_s = live_period_s
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.warmed = False
        self.live_ticks = 0

    def warmup(self) -> None:
        """Synthesise multi-day history quickly to seed the models."""
        # the virtual clock starts warmup_ticks x 900 s in the past; the live
        # phase then continues from wherever warm-up ended (≈ now)
        if self.gen.pack is None:
            self.gen.vt = time.time() - self.warmup_ticks * self.WARMUP_DT
        with self._lock:
            for _ in range(self.warmup_ticks):
                obs = self.gen.step(dt=self.WARMUP_DT, live=False)
                self.pipeline.run_tick(obs, now=self.gen.vt, training=True, dt=self.WARMUP_DT)
        self.warmed = True

    def _tick_live(self) -> dict:
        with self._lock:
            obs = self.gen.step(dt=self.window_s, live=True)
            # the virtual clock is the one the observations were stamped with
            stats = self.pipeline.run_tick(obs, now=self.gen.vt, training=False,
                                           dt=self.window_s)
            self.live_ticks += 1
        return stats

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._tick_live()
            self._stop.wait(self.live_period_s)

    def start(self) -> None:
        if not self.warmed:
            self.warmup()
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def step_once(self) -> dict:
        return self._tick_live()
