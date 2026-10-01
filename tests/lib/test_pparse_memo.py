"""P00 value policy memo (cost): ValuePolicy.apply memoises (name, value) for
payload strings; results are identical to the unmemoised policy (values,
types, modes, the extra '<name>.len'), also for repeated values."""
import numpy as np

from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import pparse as PP


def _rows(seed=0, n=600):
    r = np.random.default_rng(seed)
    pool = ["jack", "rose", "mike.w", "3fa2c9d1e8b74f2a9c0d1e2f3a4b5c6d", "application/json",
            "Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36", "a" * 80, "x9QzT1mPq7RkW2s8Lv4N",
            "550e8400-e29b-41d4-a716-446655440000", "同意", ""]
    names = ["body.kv.username", "body.kv.token", "hdr.user-agent", "hdr.x-client-ver",
             "q.kv.id", "meta.f001", "hdr.cookie", "http.route", "net.src"]
    out = []
    for _ in range(n):
        a = {}
        for nm in names:
            if r.random() < 0.7:
                a[nm] = pool[int(r.integers(0, len(pool)))]
        a["body.kv.n"] = float(r.integers(0, 5))
        a["body.keys"] = frozenset({"username", "password"})
        out.append(a)
    return out


def test_memoised_policy_is_identical_to_the_plain_one():
    cfg = EV.pconfig({})["value_policy"]
    memo, plain = PP.ValuePolicy(cfg), PP.ValuePolicy(cfg)
    plain.VMEMO_LEN = -1                       # never memoise
    for a in _rows():
        x, y = dict(a), dict(a)
        mx, my = {}, {}
        memo.apply_all(x, mx)
        plain.apply_all(y, my)
        assert x == y and mx == my
        assert {k: type(v) for k, v in x.items()} == {k: type(v) for k, v in y.items()}
    assert memo._vmemo and not plain._vmemo


def test_column_wise_learning_strata_and_uniforms_are_identical():
    """P00 learning sample (cost): the column-wise strata and the uniforms
    with a precomputed message prefix equal the per-row forms."""
    from app.engines.behavior.lib.combine import seeded_uniform
    from app.engines.raw.event_builder import learning_strata, seeded_uniforms
    r = np.random.default_rng(3)
    b = EV.BatchBuilder("oa", EV.KIND_TXN)
    for i in range(300):
        a = {"ev.ch": ["http", "tls", "dns", "l4"][i % 4], "body.len": float(r.integers(0, 4000))}
        k = i % 4
        if k == 0 and i % 8:
            a["http.route"] = f"GET oa /p{i % 5}"
        if k == 1:
            a["tls.sni"] = "mail.corp"
        if k == 2:
            a["dns.qname"] = "x.corp"
        if i % 3 == 0:
            a["net.dst"] = "10.0.0.1:443"
        b.add(1000.0 + i, f"10.0.0.{i % 7}", a)
    batch = b.build(1000.0, 1900.0)
    assert learning_strata(batch, None, None) == [EV.bootstrap_stratum(batch, i) for i in range(batch.n)]

    class H:
        @staticmethod
        def gen(a, lvl, v):
            return ("bin", int(v) // 512) if isinstance(v, float) else v
    want = [(batch.get("ev.ch", i, ""), repr(H.gen("body.len", 1, batch.get("body.len", i))))
            for i in range(batch.n)]
    assert learning_strata(batch, ("body.len", 1), H) == want
    assert seeded_uniforms("oa", batch.t1, batch.n) == [seeded_uniform("oa", batch.t1, i) for i in range(batch.n)]
