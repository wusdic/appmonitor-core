"""Round 5 (P08 cost): BindingEngine._ingest reads the batch column-wise
(strata, spec attributes, damping, times) instead of through a per-row
closure; the probe and the value histories must come out identical to
round 4's per-row form (ref_ingest below, verbatim apart from module
prefixes), including rows a quarantined / low-trust source sends, damped
rows, rows without a stratum key and spec attributes found only in evt.ctx
or as net.src."""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pytest

from helpers import T0, make_store

from app.engines.behavior import binding as BE
from app.engines.behavior.lib import m_governor as MG
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import pfd as FD
from app.engines.behavior.lib import timebins as TB
from app.engines.behavior.lib.combine import seeded_uniform


def ref_ingest(store: Any, s: str, key: str, ts_b: float, b: Any, cb: Any,
            st: Dict[str, Any], ycand: set, specs: List[Tuple[str, str]], hier: Any, tz: str,
            cal: Any, normal: Mapping[int, bool], qcache: Dict[Tuple[str, str], bool]) -> int:
    probe: FD.ProbeReservoir = st["probe"]
    rows = b.learned_rows()
    cols = [a for a in b.cols if a in ycand]
    mass = b.mass()
    # P03's outlier damping of the row (§6.9.3): a damped row (an extreme
    # outlier, a value credibly bound to another source, a who outsider)
    # never counts toward the confirmation of a new value
    asg = store.batch_at(s, EV.PAT_ASSIGN, ts_b)
    damp = asg.dense("damp", 1.0) if (asg is not None and asg.n == b.n and asg.has("damp")) else None
    dmemo: Dict[int, int] = {}
    n = 0
    dense = {a: b.dense(a) for a in list(cols) + [a for a in BE.X_ATTRS if a in b.cols]}
    for i in rows:
        i = int(i)
        ip = b.ip_of(i)
        q = qcache.get((s, ip))
        if q is None:
            # held / untrusted sources teach nothing (§6.9.3): quarantined, or
            # B28 trust below BE.TRUST_MIN (no trust row = trusted, bounded mode)
            tr_v = MG.trust(store, s, ip)
            q = qcache[(s, ip)] = bool(MG.is_quarantined(store, s, ip)
                                       or (tr_v == tr_v and tr_v < BE.TRUST_MIN))
        if q:
            continue

        def get(a: str, i: int = i) -> Any:
            if a in b.cols:
                return b.get(a, i)
            if cb is not None and a in cb.cols:
                return cb.get(a, i)
            if a == "net.src":
                return b.ip_of(i)
            return EV.ABSENT
        clean_i = damp is None or not (float(damp[i]) < 1.0)
        row = {"net.src": ip, BE.CLEAN_COL: bool(clean_i)}
        for a, arr in dense.items():
            v = arr[i]
            if v is not EV.ABSENT:
                row[a] = v
        ts = float(b.ts[i])
        if len(row) > 2:
            probe.offer(BE._stratum(get), row, float(mass[i]), ts,
                        seeded_uniform("p08", s, ts, int(b.rid[i])))
            n += 1
        if not specs or hier is None:
            continue
        day = None
        for (X, Y) in specs:
            xa, xl = FD.parse_x(X)
            ya, yl = FD.parse_x(Y)
            xv, yv = get(xa), get(ya)
            if xv is EV.ABSENT or yv is EV.ABSENT or xv is None or yv is None:
                continue
            xg = hier.gen(xa, xl, xv)
            yg = hier.gen(ya, yl, yv) if yl else yv
            if xg is None or yg is None:
                continue
            if day is None:
                mk = int(ts // 60)
                day = dmemo.get(mk)
                if day is None:
                    day = dmemo[mk] = TB.local_datetime(mk * 60.0, tz).date().toordinal()
                is_normal = bool(normal.get(day, _dt.date.fromordinal(day) not in cal.holidays))
                clean = clean_i
            pk = BE.pair_key(X, Y)
            hist = st["hist"].get(pk)
            tr = st["track"].get(pk)
            if tr is not None:
                if str(xg) not in tr:
                    continue                # history only for sources some node tracks
            elif hist is None or str(xg) not in hist.x:
                continue                    # a screened pair no sketch holds yet: its seeded sources
            if hist is None:
                hist = st["hist"][pk] = FD.ValueHistory()
            hist.observe(str(xg), yg, ts, day, is_normal, clean)
    return n


class _Hier:
    """gen(attr, level, value): level 0 the value, level 1 an IPv4 /24 prefix,
    level 3 a group label; None for a value the hierarchy rejects."""

    def gen(self, a, level, v):
        if v == "reject":
            return None
        if level == 0:
            return v
        if level == 1 and isinstance(v, str) and v.count(".") == 3:
            return v.rsplit(".", 1)[0] + ".0/24"
        if level == 3:
            return "G" + str(abs(hash(str(v)) % 1) + len(str(v)) % 3)
        return v


def _canon(o):
    if isinstance(o, float):
        return ("f", o.hex() if o == o else "nan")
    if isinstance(o, dict):
        return ("d", [(_canon(k), _canon(v)) for k, v in o.items()])
    if isinstance(o, (list, tuple)):
        return (type(o).__name__, [_canon(v) for v in o])
    if isinstance(o, np.ndarray):
        return ("nd", o.dtype.str, o.tobytes())
    if hasattr(o, "__dict__") or hasattr(type(o), "__slots__"):
        st = dict(getattr(o, "__dict__", {}))
        for cls in type(o).__mro__:
            for sl in getattr(cls, "__slots__", ()) or ():
                if hasattr(o, sl):
                    st[sl] = getattr(o, sl)
        return (type(o).__name__, [(k, _canon(v)) for k, v in st.items() if not k.startswith("_rng")])
    return (type(o).__name__, repr(o))


def _batches(seed):
    rng = np.random.default_rng(seed)
    b = EV.BatchBuilder("oa")
    c = EV.BatchBuilder("oa")
    a = EV.BatchBuilder("oa")
    users = ["jack", "rose", "mike", "mike.w", "kate", "reject"]
    n = 120
    for i in range(n):
        ip = "192.168.1.%d" % int(rng.integers(20, 30))
        at = {"net.src": ip, "ev.ch": "http" if i % 9 else None}
        r = rng.random()
        if r < 0.5:
            at["http.route"] = "POST oa /login"
            at["body.kv.username"] = users[int(rng.integers(0, len(users)))]
        elif r < 0.7:
            at["tls.sni"] = "oa.local"
            at["hdr.x-user"] = 7 if rng.random() < 0.5 else 7.0
        elif r < 0.8:
            at["net.dst"] = "10.0.0.1:443"
        if rng.random() < 0.6:
            at["client.stack"] = "s%d" % int(rng.integers(0, 3))
        if rng.random() < 0.3:
            at["sess.key"] = "k%d" % int(rng.integers(0, 4))
        b.add(T0 + 600.0 + i * 7.0, ip, at, w=float(1 + i % 3))
        c.add(T0 + 600.0 + i * 7.0, ip, {"ctx.dept": "综合部" if i % 4 else "研发"})
        a.add(T0 + 600.0 + i * 7.0, ip, {"damp": 0.1 if i % 11 == 0 else 1.0})
    bb = b.build(T0, T0 + 1500.0)
    bb.learn[::13] = False
    return bb, bb.aligned(c.build(T0, T0 + 1500.0).cols), bb.aligned(a.build(T0, T0 + 1500.0).cols)


@pytest.mark.parametrize("seed", range(4))
def test_ingest_equals_round4_per_row_form(seed, monkeypatch):
    monkeypatch.setattr(MG, "is_quarantined", lambda store, s, e, at=None: e.endswith(".27"))
    monkeypatch.setattr(MG, "trust", lambda store, s, e, at=None: 0.2 if e.endswith(".28") else float("nan"))
    b, cb, asg = _batches(seed)
    specs = [("net.src", "body.kv.username"), ("net.src@1", "body.kv.username"), ("client.stack", "hdr.x-user"),
             ("body.kv.username", "net.src"), ("sess.key", "ctx.dept"), ("net.src@3", "nope")]
    ycand = {"body.kv.username", "hdr.x-user", "http.route"}
    tz = TB.DEFAULT_TZ
    cal = TB.parse_calendar(None)
    normal = {TB.local_datetime(T0 + 600.0, tz).date().toordinal(): False}
    out = []
    for fn in ("new", "ref"):
        store = make_store()
        store.add_batch("oa", EV.PAT_ASSIGN, T0 + 1500.0, asg)
        st = {"probe": FD.ProbeReservoir(), "hist": {}, "last": {}, "bound": {}, "rebound": {},
              "track": {"net.src->body.kv.username": {"192.168.1.21", "192.168.1.22", "192.168.1.25"},
                        "client.stack->hdr.x-user": {"s0", "s1"}}}
        h = FD.ValueHistory()
        h.observe("192.168.1.0/24", "jack", T0, 1, True)
        st["hist"]["net.src@1->body.kv.username"] = h
        args = (store, "oa", "oa", T0 + 1500.0, b, cb, st, ycand, specs, _Hier(), tz, cal, normal, {})
        n = BE.BindingEngine()._ingest(*args) if fn == "new" else ref_ingest(*args)
        out.append((n, _canon(st["probe"]), _canon(st["hist"])))
        assert sum(len(hh.x) for hh in st["hist"].values()) >= 3     # histories were written
        assert st["probe"].n_rows() > 20
    assert out[0][0] == out[1][0] > 0
    assert out[0][1] == out[1][1]
    assert out[0][2] == out[1][2]
