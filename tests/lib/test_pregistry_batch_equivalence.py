"""P02 cost (round 5): AttrRegistry.observe batched per call must be
bit-identical to the per-row form it replaced.

The reference below is the per-row observe of round 4 (with the round-5
order-independent set elements, `set_elements`). The recorded stream
(tests/data/pregistry_stream_packO.pkl.gz) is every AttrRegistry call P02
and P05 made for finance, oa and crm over the first 30 h of pack O
(O-scale-500-0, seed 0): register / observe / observe_events / update_types
/ update_stats / refresh_hierarchies / check_gone / set_role /
set_value_groups with their arguments. Replaying it through the batched and
the reference observe must give the same registry, float for float.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import pickle
from typing import Any, Optional, Sequence

import numpy as np
import pytest

from app.engines.behavior.lib import pregistry as RG
from app.engines.behavior.lib import psketch as PS
from app.engines.behavior.lib.pevent import ABSENT, KIND_TXN
from app.engines.behavior.lib.phier import _ip_parse

STREAM = os.path.join(os.path.dirname(__file__), "..", "data", "pregistry_stream_packO.pkl.gz")


# ------------------------------------------------------------- reference
def ref_observe(self: RG.AttrRegistry, name: str, values: Sequence[Any], t: float, mass: Any = 1.0,
                evidence: Any = None, approx: Any = None, kind: int = KIND_TXN,
                policy: Optional[str] = None) -> int:
    """Round 4's per-row AttrRegistry.observe (set elements in set_elements order)."""
    st = self.register(name, t, kind)
    if st == "overflow":
        return 0
    rec = self.records[name]
    N = len(values)
    ms = RG._as_list(mass, N)
    es = RG._as_list(1.0 if evidence is None else evidence, N)
    aps = RG._as_list(False if approx is None else approx, N)
    vals, m, ev, apm = [], [], [], 0.0
    for v, mm, ee, aa in zip(values, ms, es, aps):
        if v is None or v is ABSENT:
            continue
        vals.append(v)
        m.append(float(mm))
        ev.append(min(1.0, float(ee)))
        if aa:
            apm += float(mm)
    n = len(vals)
    if n == 0:
        return 0
    if policy:
        rec.policy = policy
    t = float(t)
    self._roll(t)
    rec.last_seen = max(rec.last_seen, t)
    tot_m = sum(m)
    rec.pres.add(t, tot_m)
    rec.day_pres += tot_m
    if apm > 0:
        rec.approx.add(t, apm)
    te = [0.0] * len(RG._TE)
    nums, num_m = [], []
    is_time_name = rec.name.endswith("_ts") or rec.name.endswith(".ts")
    for v, e, mm in zip(vals, ev, m):
        te[0] += e
        x = RG._num(v)
        if x is not None:
            nums.append(x)
            num_m.append(mm)
            te[1] += e
            if x.is_integer():
                te[2] += e
            if is_time_name and 1e9 <= x <= 4e9:
                te[7] += e
        if isinstance(v, str):
            te[8] += e * len(v)
            if x is None:
                te[10] += e
                if _ip_parse(v) is not None:
                    te[3] += e
                elif "=" in v and len(v) <= 4096 and RG._KV_RE.match(v):
                    te[5] += e
                elif v[:1] in "{[" and len(v) <= 4096:
                    try:
                        json.loads(v)
                        te[6] += e
                    except ValueError:
                        pass
        elif isinstance(v, (frozenset, set, list)):
            te[4] += e
        elif isinstance(v, tuple):
            te[9] += e
    rec.te.add(t, te)
    card = rec.card
    for v in vals[:4096]:
        card.add(v if not isinstance(v, (frozenset, set, list, tuple))
                 else "|".join(sorted(str(x) for x in v)), t)
    top = rec.top
    key = rec.key
    dropped = rec.role_sys == "dropped"
    for v, mm, e in zip(vals, m, ev):
        top.add(key(v), t, mm, e)
        if dropped:
            continue
        if isinstance(v, (frozenset, set, list)):
            if rec.elem is None:
                rec.elem = PS.DecayedSpaceSaving(RG.TOP_K)
            for x in RG.set_elements(v)[:32]:
                rec.elem.add(str(x), t, mm, e)
    if nums and not dropped:
        if rec.num is None:
            rec.num = PS.TDigest(50.0, PS.H_M)
            rec.mom = PS.DecayedVector([PS.H_M] * 9)
        xv = np.asarray(nums)
        mv = np.maximum(np.asarray(num_m), 1e-12)
        xm = np.clip(xv, -1e30, 1e30)
        rec.num.add_many(xv, t, mv)
        pos = xv > 0
        lv = np.log(np.where(pos, xv, 1.0))
        rec.mom.add(t, np.asarray([
            mv.sum(), (mv * xm).sum(), (mv * xm ** 2).sum(), (mv * xm ** 3).sum(),
            (mv * pos).sum(), (mv * lv * pos).sum(), (mv * lv ** 2 * pos).sum(),
            (mv * lv ** 3 * pos).sum(), 0.0]))
        vmin = float(xv.min())
        rec.hier["vmin"] = min(rec.hier.get("vmin", vmin), vmin)
    return n


def ref_update_stats(self: RG.AttrRegistry, t: float) -> None:
    """Round 4's AttrRegistry.update_stats."""
    from app.engines.behavior.lib import pmdl
    for rec in self.records.values():
        items = rec.top.items(t, PS.CH_M)
        if items:
            keys, sh, other = rec.top.distribution(t, PS.CH_M)
            rec.entropy = RG.entropy_from_top(sh, other, rec.card.count(), len(keys))
            keys, sh_s, o_s = rec.top.distribution(t, PS.CH_S)
            _, sh_l, o_l = rec.top.distribution(t, PS.CH_L)
            rec.stability = 1.0 - pmdl.jsd(np.r_[sh_s, o_s], np.r_[sh_l, o_l])
        pm = float(rec.pres.read(t)[PS.CH_M])
        rec.approx_share = float(rec.approx.read(t)[0] / pm) if pm > 0 else 0.0
        if rec.mom is not None:
            mo = rec.mom.read(t)
            rec.hier["log"] = bool(RG._log_better(mo, rec.hier.get("vmin", 0.0)))


class RefRegistry(RG.AttrRegistry):
    observe = ref_observe
    update_stats = ref_update_stats


# ------------------------------------------------------------- canonical
def canon(o: Any, depth: int = 0) -> Any:
    """Exact value form: floats by their bits, arrays by dtype / shape / bytes,
    dicts in insertion order, sets sorted, objects by their slots / dict."""
    if depth > 40:
        raise RuntimeError("too deep")
    if o is None or isinstance(o, (bool, int, str, bytes)):
        return (type(o).__name__, o)
    if isinstance(o, float):
        return ("f", "nan" if o != o else o.hex())
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return canon(o.item(), depth)
    if isinstance(o, np.ndarray):
        if o.dtype == object:
            return ("ndo", o.shape, [canon(x, depth + 1) for x in o.ravel().tolist()])
        return ("nd", str(o.dtype), o.shape, np.ascontiguousarray(o).tobytes())
    if isinstance(o, (list, tuple)):
        return (type(o).__name__, [canon(x, depth + 1) for x in o])
    if isinstance(o, (set, frozenset)):
        return (type(o).__name__, sorted((canon(x, depth + 1) for x in o), key=repr))
    if isinstance(o, dict):
        return ("d", [(canon(k, depth + 1), canon(v, depth + 1)) for k, v in o.items()])
    st = {}
    if hasattr(o, "__dict__"):
        st.update(vars(o))
    for cls in type(o).__mro__:
        for sl in getattr(cls, "__slots__", ()) or ():
            if hasattr(o, sl):
                st[sl] = getattr(o, sl)
    return ("o", type(o).__name__, [(k, canon(v, depth + 1)) for k, v in st.items()])


def _replay(stream, cls):
    regs = {}
    for system, meth, args, kw in stream:
        reg = regs.get(system)
        if reg is None:
            reg = regs[system] = cls(system)
        getattr(reg, meth)(*args, **kw)
    return regs


@pytest.fixture(scope="module")
def stream():
    with gzip.open(STREAM, "rb") as f:
        return pickle.load(f)


def test_batched_observe_is_bit_identical_on_recorded_pack_o_stream(stream):
    assert sum(1 for _, m, _, _ in stream if m == "observe") > 10000
    new = _replay(stream, RG.AttrRegistry)
    ref = _replay(stream, RefRegistry)
    assert set(new) == set(ref) == {"finance", "oa", "crm"}
    for s in ref:
        a, b = new[s], ref[s]
        assert list(a.records) == list(b.records)
        for nm in b.records:
            assert canon(a.records[nm]) == canon(b.records[nm]), (s, nm)
        assert canon(vars(a)) == canon(vars(b)), s
    # the stream exercises every branch the batched form has
    recs = [r for reg in ref.values() for r in reg.records.values()]
    assert any(r.type == "text" for r in recs)
    assert any(r.type == "set" and r.elem is not None for r in recs)
    assert any(r.role_sys == "dropped" for r in recs)
    assert any(r.num is not None for r in recs)


def test_ss_add_rows_matches_sequential_adds_with_evictions_and_rescale():
    rng = np.random.default_rng(3)
    for k in (1, 3, 8):
        a = PS.DecayedSpaceSaving(k)
        b = PS.DecayedSpaceSaving(k)
        t = 1.7e9
        for step in range(40):
            # a jump far enough to force the landmark rescale on some steps
            t += float(rng.choice([60.0, 3600.0, 70 * 86400.0]))
            keys = [f"v{int(x)}" for x in rng.integers(0, 12, size=int(rng.integers(1, 30)))]
            ws = rng.random(len(keys)) * 3.0
            ws[rng.random(len(keys)) < 0.1] = 0.0
            evs = np.minimum(1.0, rng.random(len(keys)) * 1.5)
            for key, w, e in zip(keys, ws, evs):
                a.add(key, t, w, e)
            RG.ss_add_rows(b, keys, t, ws.tolist(), evs.tolist())
            assert canon(a) == canon(b)
    # other channel layouts take the generic path
    a = PS.DecayedSpaceSaving(4, mass_hl=(PS.H_M,), ev_hl=(PS.H_L,), primary=0)
    b = PS.DecayedSpaceSaving(4, mass_hl=(PS.H_M,), ev_hl=(PS.H_L,), primary=0)
    keys = ["a", "b", "c", "a", "d", "e", "f", "a"]
    for key in keys:
        a.add(key, 5.0, 1.5, 0.5)
    RG.ss_add_rows(b, keys, 5.0, [1.5] * 8, [0.5] * 8)
    assert canon(a) == canon(b)


def test_hll_items_equal_per_item_adds_and_rotation():
    a = PS.EpochHLL(p=10)
    b = PS.EpochHLL(p=10)
    items = [f"u{i}" for i in range(300)] + ["1", "1.0", "True"]
    for t in (100.0, 100.0 + 8 * 86400.0, 100.0 + 30 * 86400.0):
        for it in items:
            a.add(it, t)
        RG._hll_add_items(b, set(items), t)
        assert canon(a) == canon(b)
        items = items[::2] + [f"w{t}{i}" for i in range(50)]


def test_mixed_numeric_types_keep_their_own_keys_and_hashes():
    """1 == 1.0 == True as dict keys; the batched form must not let a memo
    merge them (their str forms, hence HLL items and top keys, differ)."""
    vals = [1, 1.0, True, "1", frozenset({1}), frozenset({1.0}), frozenset({"a", "b"}),
            frozenset({"b", "a"}), ["x", "y"], ("t", 1), float("nan"), -0.0, 0.0, None, ABSENT]
    a = RG.AttrRegistry("s")
    b = RefRegistry("s")
    for i in range(3):
        a.observe("meta.x", vals, 1e9 + i, np.linspace(0.5, 2.0, len(vals)),
                  np.linspace(0.2, 1.4, len(vals)), [j % 2 == 0 for j in range(len(vals))])
        b.observe("meta.x", vals, 1e9 + i, np.linspace(0.5, 2.0, len(vals)),
                  np.linspace(0.2, 1.4, len(vals)), [j % 2 == 0 for j in range(len(vals))])
    assert canon(vars(a)) == canon(vars(b))
    # unit evidence (the counting path) on the same values
    a.observe("meta.y", vals * 3, 2e9)
    b.observe("meta.y", vals * 3, 2e9)
    assert canon(vars(a)) == canon(vars(b))
    assert not math.isnan(a.records["meta.y"].te.read()[0])


def test_huge_values_take_the_clipped_moments():
    vals = [1e31, -2e31, 5.0, 3.5, "7", 1e300, 2.0 ** 70]
    a = RG.AttrRegistry("s")
    b = RefRegistry("s")
    for i in range(3):
        a.observe("net.x", vals[i:], 1e9 + 60 * i, 1.0 + i)
        b.observe("net.x", vals[i:], 1e9 + 60 * i, 1.0 + i)
        a.update_stats(1e9 + 60 * i)
        b.update_stats(1e9 + 60 * i)
    assert canon(vars(a)) == canon(vars(b))


def test_distributions_one_pass_equal_per_channel_reads():
    rng = np.random.default_rng(4)
    for k in (1, 5, 32):
        ss = PS.DecayedSpaceSaving(k)
        assert canon(RG._distributions(ss, 1.0, (0, 1, 2))) == canon([ss.distribution(1.0, c) for c in (0, 1, 2)])
        t = 1.7e9
        for step in range(200):
            t += float(rng.choice([10.0, 3600.0, 86400.0]))
            ss.add("v%d" % int(rng.integers(0, 3 * k)), t, float(rng.random() * 2.0), 1.0)
            if step % 37 == 0:
                got = RG._distributions(ss, t, (PS.CH_M, PS.CH_S, PS.CH_L))
                ref = [ss.distribution(t, c) for c in (PS.CH_M, PS.CH_S, PS.CH_L)]
                assert canon(got) == canon(ref)
