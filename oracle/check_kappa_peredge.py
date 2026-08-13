#!/usr/bin/env python3
"""
G6: perEdge constraint-set conditioning sweep for the WebGPU solver milestone.

For trefoil N=60/120/240/480/960: assembles the saddle matrix
K = [[A3, C^T], [C, 0]] for BOTH constraint sets (barycenter+totalLength,
barycenter+perEdge) and estimates kappa_2(K) = |lambda_max| / |lambda_min|
(K is symmetric, so its singular values equal |eigenvalues|). Mirrors the
[PREC Q2] method: power iteration for the top eigenvalue, inverse iteration
(via an LU factorization) for the bottom one; numpy.linalg.cond (exact SVD)
is used as a cross-check at N<=240 where it is cheap, per the task's own
sanity instruction. Growth exponent is fit log-log across N per mode.

Uses (imports, does not copy) the existing oracle machinery:
- tpe_stage1_oracle.assemble_inner_product / expand_vector_inner_product
  for A3 (the fractional Sobolev inner product, expanded to 3N x 3N).
- tpe_constraints_oracle's barycenter_block / total_length_block /
  edge_lengths_block / evaluate_constraint_set for the constraint rows C,
  identical to the TS ConstraintSet machinery
  (src/core/sobolev/constraintSet.ts).

Trefoil parametrization is the SAME closed-curve formula as
src/core/fixtures.ts `trefoil(n)` (verbatim transcription to numpy) so
these results describe the milestone's actual fixture, not a proxy.

Gate: spec §4 G6 — if kappa * u_f32 > 0.1 for a mode/N, that mode's solve
falls back to CPU f64 (u_f32 = 5.96e-8, single-precision unit roundoff).

@see docs/superpowers/specs/2026-08-13-webgpu-solver-design.md §4 G6
@see docs/2026-08-13-ai-research-gpu-precision.md Q2 (kappa growth-law method,
     kappa ~ N^3.3, measured 3.0e1/1.9e2/1.7e3 at N=32/64/128 on a different
     fixture)
@see docs/superpowers/plans/2026-08-13-webgpu-solver-phase0.md Task 10
"""
from __future__ import annotations

import json
import math
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.linalg as la

from tpe_stage1_oracle import (
    assemble_inner_product,
    expand_vector_inner_product,
    length_weighted_barycenter,
)
from tpe_constraints_oracle import (
    barycenter_block,
    edge_lengths,
    edge_lengths_block,
    evaluate_constraint_set,
    total_length,
    total_length_block,
)

# f32 unit roundoff (2^-24), spec §4 G6 / [PREC Q2].
U_F32 = 5.96e-8

# alpha=3, beta=6, epsilon=1e-10 — DEFAULTS (src/core/optimizer.ts:18), same
# constants the G2/toleranceSkeleton spikes use.
ALPHA = 3.0
BETA = 6.0
EPSILON = 1.0e-10

# N sweep + exact-cond cross-check cutoff. Above this, K's dense SVD (numpy
# cond) gets expensive (K is (3N+m)^2); power/inverse iteration are used
# instead — see module docstring.
N_SWEEP = [60, 120, 240, 480, 960]
EXACT_COND_MAX_N = 240


def trefoil(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Same closed-curve parametrization as src/core/fixtures.ts `trefoil(n)`
    (verbatim transcription), so K describes the milestone's real fixture.
    @see src/core/fixtures.ts
    """
    vertices = []
    edges = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        vertices.append(
            [
                math.sin(t) + 2.0 * math.sin(2.0 * t),
                math.cos(t) - 2.0 * math.cos(2.0 * t),
                -math.sin(3.0 * t),
            ]
        )
        edges.append([i, (i + 1) % n])
    return np.asarray(vertices, dtype=float), np.asarray(edges, dtype=int)


def build_K(vertices: np.ndarray, edges: np.ndarray, A3: np.ndarray, mode: str) -> np.ndarray:
    """Assemble K = [[A3, C^T], [C, 0]] for mode in {"total", "perEdge"}.
    A3 (the expensive O(N^2) fractional-Sobolev assembly) is computed ONCE
    per N by the caller and passed in — it is identical for both constraint
    modes, so re-assembling it per mode would double the N=960 wall time for
    no reason.
    """
    x0 = length_weighted_barycenter(vertices, edges)
    if mode == "total":
        L0 = total_length(vertices, edges)
        blocks = [barycenter_block(x0), total_length_block(L0)]
    elif mode == "perEdge":
        ell0 = edge_lengths(vertices, edges)
        blocks = [barycenter_block(x0), edge_lengths_block(ell0)]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    _phi, C, _counts = evaluate_constraint_set(blocks, vertices, edges)
    k = C.shape[0]
    return np.block([[A3, C.T], [C, np.zeros((k, k), dtype=float)]])


def power_iteration_lambda_max_abs(K: np.ndarray, iters: int = 300, seed: int = 0) -> float:
    """Power iteration on symmetric K: converges to the eigenvalue of
    largest |lambda| = sigma_max(K) (K symmetric => singular values are
    |eigenvalues|). [PREC Q2] method.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(K.shape[0])
    x /= np.linalg.norm(x)
    lam = 0.0
    for _ in range(iters):
        y = K @ x
        ny = np.linalg.norm(y)
        if ny < 1e-300:
            break
        x = y / ny
        lam = float(x @ (K @ x))
    return abs(lam)


def inverse_iteration_lambda_min_abs(K: np.ndarray, iters: int = 300, seed: int = 1) -> float:
    """Inverse iteration on symmetric K via an LU factorization (scipy
    lu_factor/lu_solve): the Rayleigh quotient of K^-1 converges to its
    largest-magnitude eigenvalue, 1/lambda_min(K). [PREC Q2] method.
    """
    fac = la.lu_factor(K)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(K.shape[0])
    x /= np.linalg.norm(x)
    mu = 0.0
    for _ in range(iters):
        y = la.lu_solve(fac, x)
        ny = np.linalg.norm(y)
        if ny < 1e-300:
            break
        x = y / ny
        mu = float(x @ la.lu_solve(fac, x))
    if mu == 0.0:
        return math.inf
    return abs(1.0 / mu)


def estimate_kappa(K: np.ndarray, n: int) -> Tuple[float, str]:
    """kappa_2(K) via power/inverse iteration, cross-checked against
    numpy.linalg.cond (exact SVD) at N <= EXACT_COND_MAX_N. Returns
    (kappa, method) where method records which estimate is REPORTED.
    """
    lam_max = power_iteration_lambda_max_abs(K)
    lam_min = inverse_iteration_lambda_min_abs(K)
    kappa_iter = lam_max / lam_min if lam_min > 0 else math.inf

    if n <= EXACT_COND_MAX_N:
        kappa_exact = float(np.linalg.cond(K))
        ratio = kappa_iter / kappa_exact if kappa_exact > 0 else math.inf
        print(
            f"    cross-check: power/inverse iteration kappa={kappa_iter:.4e} "
            f"vs numpy.linalg.cond kappa={kappa_exact:.4e} (ratio {ratio:.4f})"
        )
        return kappa_exact, "cond(exact SVD); power/inverse iteration cross-checked"
    return kappa_iter, "power/inverse iteration"


def fit_growth_exponent(ns: List[int], kappas: List[float]) -> float:
    """log-log linear fit: kappa ~ C * N^p, returns p."""
    x = np.log(np.asarray(ns, dtype=float))
    y = np.log(np.asarray(kappas, dtype=float))
    p, _c = np.polyfit(x, y, 1)
    return float(p)


def main(argv: List[str]) -> int:
    rows: List[Dict[str, Any]] = []
    by_mode: Dict[str, Tuple[List[int], List[float]]] = {"total": ([], []), "perEdge": ([], [])}
    flagged: List[Tuple[str, int, float]] = []

    print(f"{'N':>5} {'mode':>8} {'kappa':>14} {'kappa*u_f32':>14} {'method':>45} {'sec':>7}")
    for n in N_SWEEP:
        vertices, edges = trefoil(n)
        t_assemble0 = time.time()
        # Shared A3 assembly (the O(N^2) expensive part) computed ONCE per N,
        # reused for both constraint modes below.
        t_a0 = time.time()
        _B, _B0, A, _info = assemble_inner_product(vertices, edges, ALPHA, BETA, EPSILON)
        A3 = expand_vector_inner_product(A)
        t_assemble_a3 = time.time() - t_a0
        print(f"    (N={n} A3 assembly: {t_assemble_a3:.2f}s, shared by both modes)")
        for mode in ("total", "perEdge"):
            t0 = time.time()
            K = build_K(vertices, edges, A3, mode)
            t_build = time.time() - t0
            t1 = time.time()
            kappa, method = estimate_kappa(K, n)
            t_est = time.time() - t1
            kappa_u = kappa * U_F32
            rows.append({"n": n, "mode": mode, "kappa": kappa, "kappaTimesUf32": kappa_u})
            by_mode[mode][0].append(n)
            by_mode[mode][1].append(kappa)
            if kappa_u > 0.1:
                flagged.append((mode, n, kappa_u))
            print(
                f"{n:>5} {mode:>8} {kappa:>14.4e} {kappa_u:>14.4e} {method:>45} "
                f"{t_build + t_est:>7.2f}"
            )
        print(f"    (N={n} total wall time: {time.time() - t_assemble0:.2f}s)")

    print()
    exponents: Dict[str, float] = {}
    for mode, (ns, kappas) in by_mode.items():
        p = fit_growth_exponent(ns, kappas)
        exponents[mode] = p
        print(f"fitted growth exponent, mode={mode}: kappa ~ N^{p:.3f}")

    print()
    if flagged:
        print("G6 GATE: kappa*u_f32 > 0.1 flagged for:")
        for mode, n, kappa_u in flagged:
            print(f"  mode={mode} N={n}: kappa*u_f32={kappa_u:.4e} > 0.1 -> falls back to CPU f64 solve")
    else:
        print("G6 GATE: no mode/N exceeded kappa*u_f32 > 0.1")

    out_path = "bench/results/2026-08-13-gpu-phase0-g6.json"
    out = {
        "gate": "G6",
        "rows": rows,
        "growthExponent": exponents,
        "uF32": U_F32,
        "flagged": [{"mode": m, "n": n, "kappaTimesUf32": k} for m, n, k in flagged],
        "exactCondMaxN": EXACT_COND_MAX_N,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
