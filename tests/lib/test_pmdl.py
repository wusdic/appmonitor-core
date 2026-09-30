"""lib/pmdl.py: code lengths, entropies, dependencies and bounds."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.engines.behavior.lib import pmdl as M


def test_ml_code_length_is_minimal_over_fixed_codes():
    c = np.array([5.0, 3.0, 0.0, 2.0])
    ml = M.ml_code_length(c)
    for p in (np.full(4, 0.25), np.array([0.4, 0.3, 0.1, 0.2]), np.array([0.7, 0.1, 0.1, 0.1])):
        fixed = float(-(c * np.log2(p)).sum())
        assert ml <= fixed + 1e-12
    assert ml == pytest.approx(10 * M.entropy_plugin(c))
    rows = M.ml_code_length_rows(np.stack([c, c * 2]))
    assert rows[1] == pytest.approx(2 * rows[0])


def test_kt_equals_sequential_predictive():
    seq = [0, 1, 0, 0, 2, 0, 1]
    K = 3
    counts = np.zeros(K)
    L = 0.0
    for x in seq:
        L += -math.log2((counts[x] + 0.5) / (counts.sum() + 0.5 * K))
        counts[x] += 1
    assert M.kt_code_length(counts) == pytest.approx(L)
    assert M.dirichlet_code_length(counts, 0.5) == pytest.approx(L)


def test_dirichlet_predictive_backs_off_and_is_evidence_scaled():
    parent = np.array([0.5, 0.5])
    p0 = M.dirichlet_predictive([1.0, 0.0], 0.0, 2.0, parent)
    assert np.allclose(p0, parent)
    p_small = M.dirichlet_predictive([1.0, 0.0], 2.0, 2.0, parent)
    p_big = M.dirichlet_predictive([1.0, 0.0], 200.0, 2.0, parent)
    assert p_small[0] == pytest.approx(0.75) and p_big[0] > 0.99
    assert p_small.sum() == pytest.approx(1.0)


def test_entropies_and_divergences():
    u = np.ones(8)
    assert M.entropy_plugin(u) == pytest.approx(3.0)
    assert M.entropy_miller_madow(u * 10) > 3.0
    assert M.entropy_chao_shen(u * 10) == pytest.approx(3.0, abs=0.05)
    assert M.jsd([1, 0], [0, 1]) == pytest.approx(1.0)
    assert M.jsd([1, 2, 3], [2, 4, 6]) == pytest.approx(0.0, abs=1e-12)
    joint = np.array([[10, 0], [0, 10]])
    assert M.mutual_information(joint) == pytest.approx(1.0)
    assert M.cond_entropy(joint) == pytest.approx(0.0)
    assert M.g3(joint) == 0.0
    assert M.g3(np.array([[5, 5], [0, 10]])) == pytest.approx(0.25)


def test_bounds():
    assert M.hoeffding_bound(1.0, 100, 0.05) == pytest.approx(math.sqrt(math.log(20) / 200))
    eb = M.empirical_bernstein_bound(0.0, 1.0, 100, 1e-4)
    assert eb == pytest.approx(3 * math.log(3e4) / 100)
    assert sum(M.time_uniform_delta(0.1, k) for k in range(1, 100000)) < 0.1
    assert M.jeffreys_lower(9, 9) > 0.8 > M.jeffreys_lower(8, 8) - 0.1
    assert M.good_turing_unseen(0, 0, 99) == pytest.approx(0.005)
    assert M.poisson_zero_p(3.0) == pytest.approx(math.exp(-3))
    assert M.sidak(1e-3, 10) == pytest.approx(1 - 0.999 ** 10)
    assert M.hdr_p([0.5, 0.3, 0.2], 2) == pytest.approx(0.1)
    assert M.hdr_p([0.5, 0.3, 0.2], 0) == pytest.approx(0.75)
    assert M.log2_mean_exp2([10, 10]) == pytest.approx(10)
    assert M.log2_mean_exp2([1000.0, -1000.0]) == pytest.approx(999.0)


def test_leave_one_out_binding_prior_numbers():
    """§6.12: two other members with 5 pure logins each; an IP with 5 pure
    logins has LB ~ 0.88 (>= 0.8); a per-IP Jeffreys bound needs 9 pure logins
    to reach 0.8. Measured: "LB >= 0.9 from the 6th" holds when all three
    members are at 6 (0.8998); with the others at 5 it is reached at the 8th."""
    a0, b0 = M.eb_beta_prior([5, 5], [5, 5])
    lb5 = M.beta_quantile(0.05, a0 + 5, b0)
    assert lb5 >= 0.8 and lb5 == pytest.approx(0.8812, abs=1e-3)
    assert M.beta_quantile(0.05, a0 + 8, b0) >= 0.9 > M.beta_quantile(0.05, a0 + 7, b0)
    a6, b6 = M.eb_beta_prior([6, 6], [6, 6])
    assert M.beta_quantile(0.05, a6 + 6, b6) == pytest.approx(0.8998, abs=1e-3)
    assert M.jeffreys_lower(8, 8) < 0.8 <= M.jeffreys_lower(9, 9)
    assert M.eb_beta_prior([5], [5]) == (0.5, 0.5)


def test_description_lengths():
    assert M.split_description_length(6, 2, 256) == pytest.approx(math.log2(6) + 16)
    seen = M.ip_two_part_code(0.25, 0.0, True, 0.1, 32)
    unseen = M.ip_two_part_code(0.0, 0.0, False, 0.1, 32)
    assert seen == pytest.approx(2.0) and unseen == pytest.approx(-math.log2(0.1) + 32)
