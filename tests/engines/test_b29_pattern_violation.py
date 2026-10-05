"""B29 (evaluator round 4): an incident opened by a P03 pattern violation is
explained first by the violated constraint of the learned pattern."""
from __future__ import annotations

from types import SimpleNamespace

from app.core.store import MetricStore
from app.engines.behavior.explain import ExplainEngine
from app.models.schema import BehaviorEvent, Severity


def test_the_violated_constraint_is_the_first_reason():
    """Pack O (rounds 2-3): the numeric attributions of P03-opened incidents
    were empty under the progressive decision chain and the explanation fell
    back to the request count ('操作次数 n/a，常态 0–58'); PG6's 'B29 top reason
    = violated constraint' was 0 on every seed."""
    st = MetricStore()
    eid = st.add_event(BehaviorEvent(
        system="oa", entity="192.168.1.21", ts=1000.0, kind="pattern_violation", score=0.3,
        severity=Severity.HIGH, p_value=0.02, axes=["credential"],
        extra={"type": "content", "observed": "body.kv.username", "expected": "cross_binding",
               "p": 0.02, "p_day": 0.4, "flags": ["cross_binding"], "pattern_id": "p:oa:0:18@1.0",
               "route": "POST oa /login",
               "statement_zh": "192.168.1.21 执行 POST /login：body.kv.username 与已学到的绑定不符",
               "statement_en": "192.168.1.21 performed POST /login: body.kv.username contradicts a learned binding"}))
    inc = SimpleNamespace(
        system="oa", entity="192.168.1.21", opened=1000.0, severity=Severity.HIGH, axes=["credential"],
        e_day_min=0.0003,
        evidence=[{"ts": 1000.0, "source": "b27", "state": "open"},
                  {"ts": 1000.0, "source": "event", "kind": "pattern_violation", "event_id": eid,
                   "severity": "high"}])
    a = ExplainEngine._violation_attr(st, inc, 1900.0)
    assert a is not None and a["feature"] == "conf_content" and a["constraint"] == "body.kv.username"
    from app.eval.metrics import top_features
    expl = {"attributions": [a, {"feature": "intensity", "range": "0–58", "observed_text": "n/a"}]}
    assert top_features(expl, 1) == ["conf_content"]
    zh, en, _parts = ExplainEngine._narrative(inc, dict(expl, new_tokens=[], vanished=[]), "Wed", "周三")
    assert "绑定不符" in zh and "0–58" not in zh.split("\n")[0]
