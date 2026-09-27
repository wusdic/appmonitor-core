"""Offline generator of evt.BEACON_NULL_LR (B12 Gamma renewal LR null table).

Prints the constant block to paste into backend/app/engines/behavior/lib/evt.py
and, on stderr, the evidence that it is right:

  1. exact quantiles: Bromwich inversion of the closed-form Mellin transform of
     T = n (ln mean x - mean ln x) under i.i.d. Exp intervals (evt module
     docstring), mapped through the runtime LR(s) code;
  2. Monte-Carlo check (>= 2e4 draws per n, default 1e5): realised
     P0(LR >= q) / nominal for every column the sample can resolve (>= 10
     expected exceedances), with a binomial z-score;
  3. importance-sampling check of the tail (1e-6, 1e-9, 1e-12): x_i ~ Gamma(k)
     i.i.d. is the Dirichlet(k) tilt of the null, whose likelihood ratio is a
     function of T only, w = G(n) G(k)^n / G(nk) exp((k - 1)(T + n ln n)),
     with k chosen so that E_k[T] sits at the target quantile;
  4. interpolation error of beacon_null_p against the exact log p on a dense
     LR grid, and monotonicity of the rows in n (the conservative row rule).

Usage:
    .venv/bin/python scripts/gen_beacon_null.py [--sims 100000] [--is-sims 20000]
        [--seed 20260927] > /tmp/beacon_null_block.py
Runtime ~30 s on one core. Deterministic given the seed.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
from scipy import optimize, special

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.engines.behavior.lib import evt as E  # noqa: E402

LN10 = math.log(10.0)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def s_of_lr(lr: float, n: int) -> float:
    """Inverse of the (decreasing) runtime map s -> LR on kappa_hat > 1."""
    s_one = float(np.euler_gamma)                   # kappa = 1 <=> s = gamma
    return optimize.brentq(lambda s: E._kappa_lr_scalar(s, n)[1] - lr, 1e-12, s_one,
                           xtol=1e-15, rtol=1e-14, maxiter=300)


def mc_check(table: np.ndarray, n_sims: int, seed: int) -> float:
    log(f"-- Monte-Carlo check: {n_sims} null draws per n (realised/nominal, z) --")
    probs = 10.0 ** np.asarray(E.BEACON_NULL_LOG10P)
    worst = 0.0
    for i, n in enumerate(E.BEACON_NULL_N):
        lr = E._mc_null_lr(n, n_sims, np.random.default_rng([seed, n]))
        cells = []
        for j, p in enumerate(probs):
            if p * n_sims < 10:
                continue
            hits = int(np.sum(lr >= table[i, j]))
            z = (hits - p * n_sims) / math.sqrt(n_sims * p * (1 - p))
            worst = max(worst, abs(z))
            cells.append(f"{p:.0e}:{hits / (p * n_sims):.2f}({z:+.1f})")
        log(f"n={n:3d} P(LR>0)={np.mean(lr > 0):.3f}  " + " ".join(cells))
    return worst


def is_tail_prob(t: float, n: int, n_draws: int, rng: np.random.Generator):
    """Importance-sampling estimate of P0(T <= t) and its relative s.e."""
    kk = optimize.brentq(lambda k: n * (special.psi(n * k) - math.log(n) - special.psi(k)) - t,
                         1.0 + 1e-9, 1e9)
    x = rng.gamma(kk, size=(n_draws, n))
    tt = n * E._log_ratio_stat(x)
    logw = (math.lgamma(n) + n * math.lgamma(kk) - math.lgamma(n * kk)
            + (kk - 1.0) * (tt + n * math.log(n)))
    w = np.where(tt <= t, np.exp(logw - logw.max()), 0.0)
    est = w.mean()
    return math.log(est) + logw.max(), w.std() / est / math.sqrt(n_draws)


def is_check(table: np.ndarray, n_draws: int, seed: int) -> float:
    log(f"-- importance-sampling tail check: {n_draws} tilted draws per cell "
        "(log10 p_IS - log10 p_nominal +- s.e.) --")
    worst = 0.0
    cols = [E.BEACON_NULL_LOG10P.index(v) for v in (-6.0, -9.0, -12.0)]
    for i, n in enumerate(E.BEACON_NULL_N):
        rng = np.random.default_rng([seed, n, 7])
        cells = []
        for j in cols:
            t = n * s_of_lr(float(table[i, j]), n)
            lp, rse = is_tail_prob(t, n, n_draws, rng)
            d = lp / LN10 - E.BEACON_NULL_LOG10P[j]
            worst = max(worst, abs(d) / max(rse / LN10, 1e-3))
            cells.append(f"{E.BEACON_NULL_LOG10P[j]:.0f}:{d:+.4f}+-{rse / LN10:.4f}")
        log(f"n={n:3d} " + " ".join(cells))
    return worst


def interp_check(table: np.ndarray) -> tuple:
    """(max conservative, max anti-conservative) log10 error of beacon_null_p
    between the first and last knots. log p is convex in LR, so the chords
    of the linear interpolation lie above it: errors should be conservative."""
    log("-- beacon_null_p interpolation vs exact: max log10(p_interp / p_exact), "
        "max log10(p_exact / p_interp) --")
    knots = E._build_knots(table)
    saved = E._NULL_KNOTS
    E._NULL_KNOTS = knots
    cons = anti = 0.0
    try:
        for i, n in enumerate(E.BEACON_NULL_N):
            lrs = np.linspace(table[i, 0], table[i, -1], 97)[1:-1]
            c_n = a_n = 0.0
            for lr in lrs:
                exact = E._null_log_cdf_T(n * s_of_lr(float(lr), n), n) / LN10
                d = math.log10(E.beacon_null_p(lr, n)) - exact
                c_n, a_n = max(c_n, d), max(a_n, -d)
            cons, anti = max(cons, c_n), max(anti, a_n)
            log(f"n={n:3d} conservative {c_n:.1e}  anti-conservative {a_n:.1e}")
    finally:
        E._NULL_KNOTS = saved
    return cons, anti


def emit(table: np.ndarray) -> str:
    lines = ["BEACON_NULL_LR: np.ndarray = np.array(["]
    for n, row in zip(E.BEACON_NULL_N, table):
        vals = [f"{v:.6f}" for v in row]
        chunks = [", ".join(vals[a:a + 7]) for a in range(0, len(vals), 7)]
        body = (",\n     ").join(chunks)
        lines.append(f"    [{body}],  # n={n}")
    lines.append("])")
    lines.append("BEACON_NULL_LR.setflags(write=False)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sims", type=int, default=100_000, help="MC draws per n (>= 20000)")
    ap.add_argument("--is-sims", type=int, default=20_000, help="IS draws per tail cell")
    ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args()
    if args.sims < 20_000:
        ap.error("--sims must be >= 20000")

    t0 = time.time()
    table = E._exact_null_lr_table(E.BEACON_NULL_N, E.BEACON_NULL_LOG10P)
    log(f"exact table {table.shape} in {time.time() - t0:.1f} s")
    assert np.all(np.isfinite(table))
    assert np.all(np.diff(table, axis=1) > 0), "rows must increase as p decreases"
    mono_n = bool(np.all(np.diff(table, axis=0) < 0))
    log(f"rows strictly decreasing in n (conservative row rule): {mono_n}")

    z_mc = mc_check(table, args.sims, args.seed)
    z_is = is_check(table, args.is_sims, args.seed)
    e_cons, e_anti = interp_check(table)
    log(f"summary: max |z| MC = {z_mc:.2f}, max IS dev / s.e. = {z_is:.2f}, interpolation "
        f"log10 error conservative <= {e_cons:.1e}, anti-conservative <= {e_anti:.1e}, "
        f"total {time.time() - t0:.0f} s")
    if not mono_n or z_mc > 4.5 or z_is > 4.5 or e_cons > 0.05 or e_anti > 0.005:
        log("CHECK FAILED")
        return 1
    print(emit(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
