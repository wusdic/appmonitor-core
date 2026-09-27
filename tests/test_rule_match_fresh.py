"""lib-4 RuleMatchEngine matches only on values written at the current tick.

A match answers "what is this entity doing right now". Evaluating the latest
value of any age made a host that uploaded once keep matching bulk_upload
(HIGH) on every idle tick afterwards; in the eval packs the nightly backup
hosts matched on ~150 of 152 ticks, which zeroed their warm-up trust (B28:
training trust is 0 while a lib-4 match >= HIGH exists) and fed B26 / B28
attacker evidence from the first live tick on.
"""
from __future__ import annotations

from helpers import DT, T0, add_obs_tick, make_store, run_engine

from app.engines.signature.rule_match import RuleMatchEngine
from app.engines.signature.store import Signature, SignatureStore

S, E = "erp", "10.0.9.5"


def _engine() -> RuleMatchEngine:
    sigs = SignatureStore()
    sigs.signatures.append(Signature(
        id="bulk_upload", label="bulk upload", category="transfer", severity="high",
        all=[{"metric": "l4.bytes_up", "op": "gt", "value": 3e6}]))
    return RuleMatchEngine(sigs)


def test_matches_only_on_fresh_values():
    store, eng = make_store(), _engine()
    add_obs_tick(store, S, E, T0, {"l4.bytes_up": 2e7})
    run_engine(eng, store, T0)
    assert [m.signature_id for m in store.matches(S, E, limit=10)] == ["bulk_upload"]
    for k in range(1, 5):                          # idle: nothing new observed
        run_engine(eng, store, T0 + k * DT)
    assert len(store.matches(S, E, limit=10)) == 1
    add_obs_tick(store, S, E, T0 + 5 * DT, {"l4.bytes_up": 1e3})
    run_engine(eng, store, T0 + 5 * DT)            # active again, below the threshold
    assert len(store.matches(S, E, limit=10)) == 1
