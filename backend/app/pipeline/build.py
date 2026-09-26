"""Assembly: build the store, the full engine registry, and the runtime.

This is the one place that knows the concrete engine set. Everything else
depends only on the Pipeline / Registry / Store abstractions.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

import yaml

from ..core.engine import Registry
from ..core.store import MetricStore
from ..engines.behavior.anomaly import AnomalyEngine
from ..engines.behavior.baseline import BaselineEngine
from ..engines.behavior.clustering import ClusteringEngine
from ..engines.behavior.drift import DriftEngine
from ..engines.behavior.feature_vector import FeatureVectorEngine
from ..engines.behavior.fingerprint import FingerprintEngine
from ..engines.behavior.sequence import SequenceEngine
from ..engines.derived.aggregation import AggregationEngine
from ..engines.derived.entropy import EntropyEngine
from ..engines.derived.graph import GraphEngine
from ..engines.derived.periodicity import PeriodicityEngine
from ..engines.derived.ratio import RatioEngine
from ..engines.derived.session import SessionEngine
from ..engines.derived.trend import TrendEngine
from ..engines.raw.active_probe import ActiveProbeEngine
from ..engines.raw.dns import DNSEngine
from ..engines.raw.http import HTTPEngine
from ..engines.raw.l2l3 import L2L3Engine
from ..engines.raw.l4flow import L4FlowEngine
from ..engines.raw.tls import TLSEngine
from ..engines.signature.correlation import CorrelationEngine
from ..engines.signature.rule_match import RuleMatchEngine
from ..engines.signature.store import SignatureStore
from .generator import TrafficGenerator
from .orchestrator import Pipeline

DATA_DIR = os.environ.get(
    "APPMON_SIGNATURE_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "signatures"))


def build_registry(sig_store: SignatureStore, composite_rules) -> Registry:
    reg = Registry()
    # raw (原始指标库)
    reg.add(L2L3Engine(), L4FlowEngine(), HTTPEngine(), TLSEngine(),
            DNSEngine(), ActiveProbeEngine())
    # derived (次生指标库)
    reg.add(AggregationEngine(), RatioEngine(), PeriodicityEngine(),
            EntropyEngine(), SessionEngine(), GraphEngine(), TrendEngine())
    # behaviour (行为库) — heavy ML engines stride via `interval`
    reg.add(FeatureVectorEngine(),
            BaselineEngine(interval=4),
            FingerprintEngine(interval=4),
            ClusteringEngine(interval=8),
            AnomalyEngine(interval=2),
            DriftEngine(),
            SequenceEngine())
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
    in a background thread. The API reads the store this exposes."""

    def __init__(self, window_s: int = 60, warmup_ticks: int = 180, live_period_s: float = 3.0):
        self.store = MetricStore()
        self.sig_store, self.composite_rules = load_signatures()
        self.registry = build_registry(self.sig_store, self.composite_rules)
        self.pipeline = Pipeline(self.store, self.registry, window_s=window_s)
        self.gen = TrafficGenerator(window_s=window_s)
        self.window_s = window_s
        self.warmup_ticks = warmup_ticks
        self.live_period_s = live_period_s
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.warmed = False
        self.live_ticks = 0

    def warmup(self) -> None:
        """Synthesise multi-day history quickly to seed baselines/fingerprints."""
        # start the virtual clock several days in the past, step 15 min per tick
        self.gen.vt = time.time() - self.warmup_ticks * 900
        for _ in range(self.warmup_ticks):
            obs = self.gen.step(dt=900, live=False)
            self.pipeline.run_tick(obs, now=self.gen.vt, training=True)
        # align virtual clock to now for the live phase
        self.gen.vt = time.time()
        self.warmed = True

    def _loop(self) -> None:
        while not self._stop.is_set():
            obs = self.gen.step(dt=self.window_s, live=True)
            self.pipeline.run_tick(obs, now=time.time())
            self.live_ticks += 1
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
        obs = self.gen.step(dt=self.window_s, live=True)
        stats = self.pipeline.run_tick(obs, now=time.time())
        self.live_ticks += 1
        return stats
