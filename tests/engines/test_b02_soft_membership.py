"""Round 4: B02 soft membership P(c | e) is the posterior of an exponential
distance model per role (scale d90_c / ln 10, normalised density), not the
unnormalised exp(-D / d90_c) of v2, which let a diffuse role take mass from a
tight one (pack A's new employee L8: p = 0.66 < 0.8 for the human role it
sat well inside; linking-pack runs: 0.66 -> 0.91)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.engines.behavior import peer_group as PG


def _v2(D, d90):
    lg = -np.asarray(D) / np.asarray(d90)
    p = np.exp(lg - lg.max())
    return p / p.sum()


def test_tight_role_member_is_typed_confidently():
    D, d90 = [0.08, 0.32], [0.15, 0.27]
    p = PG.soft_membership(np.array([D]), d90)[0]
    assert p[0] >= 0.85
    assert _v2(D, d90)[0] < 0.8                  # the v2 weights (regression reference)


def test_posterior_properties():
    # symmetric roles, equidistant: 1/2 each; rows sum to 1; farther -> less
    p = PG.soft_membership(np.array([[0.2, 0.2], [0.1, 0.4], [0.4, 0.1]]), [0.15, 0.15])
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p[0, 0] == pytest.approx(0.5)
    assert p[1, 0] > 0.99 and p[2, 1] > 0.99
    # the density's normaliser: at equal D a tighter role is the likelier source
    # when D is small, the diffuse one when D is large
    q = PG.soft_membership(np.array([[0.05, 0.05], [0.8, 0.8]]), [0.15, 0.6])
    assert q[0, 0] > 0.5 and q[1, 1] > 0.5


def test_d90_is_the_ninetieth_percentile_of_the_model():
    # members' distances drawn from the model: 90 % fall within d90
    rng = np.random.default_rng(0)
    d90 = 0.2
    x = rng.exponential(d90 / math.log(10.0), 20000)
    assert np.mean(x <= d90) == pytest.approx(0.9, abs=0.01)


# ------------------------------------------------ level-1 family descriptor
def _desc(i: int, machine: bool, rng) -> "PG._Desc":
    """A member of a machine (API client) or human role; each individual uses
    its own hosts / endpoints (system-specific, individual), the channel and
    method mix is the role's."""
    sysn = ("erp", "oa", "api")[i % 3]
    d = PG._Desc(key=f"{sysn}|10.0.{int(machine)}.{i}", s=sysn, e=f"10.0.{int(machine)}.{i}")
    d.A = (0.7 if machine else 0.2) + rng.normal(0, 0.03)
    d.clr = np.array([1.0, -0.5, 0.3, -0.8]) + rng.normal(0, 0.05, 4)
    host = f"{sysn}-api.corp" if machine else f"{sysn}.corp"
    segs = [f"ep{i}{k}" for k in range(3)]
    mix = {"http|write": 0.5, "http|read": 0.3, "tls|read": 0.2} if machine else \
        {"http|read": 0.7, "http|write": 0.1, "tls|read": 0.2}
    fam = {}
    for ch, w in mix.items():
        if ch.startswith("http"):
            for sg in segs:
                fam[f"{ch}|{host}|{sg}"] = w / len(segs)
        else:
            fam[f"{ch}|corp|"] = w
    d.fam = fam
    s48 = np.ones(48) if machine else np.r_[np.zeros(16), np.ones(20), np.zeros(12)]
    d.s48 = s48 / s48.sum()
    d.dev = "library" if machine else "browser"
    return d


def test_role_distance_ignores_individual_hosts_and_endpoints():
    rng = np.random.default_rng(1)
    ds = [_desc(i, True, rng) for i in range(6)] + [_desc(i, False, rng) for i in range(6)]
    D = PG.role_distance(ds)
    within = np.r_[D[:6, :6][~np.eye(6, dtype=bool)], D[6:, 6:][~np.eye(6, dtype=bool)]]
    between = D[:6, 6:].ravel()
    # the individual families no longer push role members towards ROLE_EPS
    assert within.max() < 0.5 * PG.ROLE_EPS
    assert between.min() > 2 * PG.ROLE_EPS
    lab, fb = PG.level1(D)
    assert not fb and len(set(lab[:6])) == 1 and len(set(lab[6:])) == 1 and lab[0] != lab[6]
    # the v2 fine families: same-role pairs at ~the merge radius
    fine = PG._sqrt_jsd(PG._dense([d.fam for d in ds]), PG._dense([d.fam for d in ds]))
    assert np.median(fine[:6, :6][~np.eye(6, dtype=bool)]) > 0.5


# ------------------------------------------------ level 2 and B15 separability
def test_level2_groups_split_by_identity_confusability():
    ds = [PG._Desc(key=f"erp|{e}", s="erp", e=e) for e in ("a", "b", "c", "d", "x")]
    idm = {"erp": {"stats": {"a": {"confusable_with": ["b"]}, "b": {"confusable_with": []},
                             "c": {"confusable_with": []}, "d": {"confusable_with": []}}}}
    lab = np.array([0, 0, 0, 0, 0])                 # HDBSCAN put all five together
    out = PG.split_by_identity(lab, ds, idm)
    assert out[0] == out[1]                          # confusable (B15): one sub-class
    assert len({out[0], out[2], out[3]}) == 3        # separable individuals: their own
    assert out[4] == 0                               # not enrolled: HDBSCAN's grouping
    # without an identity model nothing changes
    assert list(PG.split_by_identity(lab, ds, {"erp": None})) == list(lab)
