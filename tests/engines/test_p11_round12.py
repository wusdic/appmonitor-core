"""P11 who groups, round 12 (views / groups / bindings owner): base communities
are checked member by member (who_groups._peel)."""
from __future__ import annotations

import numpy as np

from app.engines.behavior import who_groups as WG

K = 64


def _rows(spec, seed=0):
    """MinHash rows: spec = [(base, share)] - each row agrees with its base row
    on `share` of the K positions (the rest are fresh values)."""
    rng = np.random.default_rng(seed)
    bases = {}
    out = []
    for b, share in spec:
        if b not in bases:
            bases[b] = rng.integers(1, 2 ** 62, size=K, dtype=np.uint64)
        r = bases[b].copy()
        k = int(round((1.0 - share) * K))
        r[:k] = rng.integers(1, 2 ** 62, size=k, dtype=np.uint64)
        out.append(r)
    return np.vstack(out)


def test_a_weakly_linked_member_is_peeled_off_its_community():
    """O-real seed 2 (round 11, day 32): 综合部's approver 192.168.1.21 had
    mutual-kNN edges only to 财务部's clerks (J ~0.3; the clerks with each
    other ~0.86) and Louvain put it into 财务部's group, so 综合部's parts of
    the family login named finance's addresses. A member less similar to the
    others than RHO x their own similarity leaves the community."""
    # rows 0-1: the clerks (agree on 0.9 of a shared base); row 2: the approver
    # (0.3 of the clerks' base); rows 3-5: a department of three alike
    M = _rows([("fin", 0.92), ("fin", 0.92), ("fin", 0.3), ("ga", 0.8), ("ga", 0.8), ("ga", 0.8)])
    lab = np.array([0, 0, 0, 3, 3, 3])
    intra = {0: 0.4, 3: 0.65}
    out = WG._peel(lab, M, WG.RHO, intra=intra)
    assert out[0] == out[1] == 0 and out[2] not in (0, 3)
    assert out[3] == out[4] == out[5] == 3                      # alike members stay
    assert intra[0] > 0.7                                        # the clerks' cohesion, recomputed


def test_identical_and_uniform_communities_are_kept():
    """A pool of identical signatures and a loose but uniform population are
    left as they are (every member as similar to the rest as the rest are)."""
    M = _rows([("p", 1.0)] * 4 + [("u", 0.5)] * 6, seed=1)
    lab = np.array([0] * 4 + [4] * 6)
    out = WG._peel(lab, M, WG.RHO)
    assert list(out) == list(lab)
