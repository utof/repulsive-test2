#!/usr/bin/env python3
"""Numpy twin of the Phase 2a saddle reformulation — [DESIGN] 4.1 steps 1-3.

WHY THIS FILE EXISTS: it is the reason gate CR can be trusted at all.  CR
(`test/gpu/cr.test.ts`) compares one TypeScript implementation of the shift +
Schur + Woodbury reformulation against the core's own `solveSaddleFromA`.  A
single implementation cannot distinguish "the TypeScript is wrong" from "the
ALGEBRA in the spec is wrong", and those two have opposite remedies.  This twin
is an INDEPENDENT implementation of the same three steps, in numpy/LAPACK, over
the same fixture and the same right-hand sides:

  * CR red, twin green  -> the defect is in `src/gpu/saddle.ts`.
  * CR red, twin red    -> the defect is in the algebra of [DESIGN] 4.1, and the
                           SPEC is what has to change.  No GPU work starts
                           either way ([DESIGN] 6).

It is deliberately NOT transcribed from the plan: a twin copied from the same
source as the thing it checks inherits that source's errors and localises
nothing.  It follows the spec's four-line description of steps 1-3 and is
written against numpy primitives (`scipy.linalg.cho_factor` / `cho_solve`,
`np.linalg.solve`) rather than the hand-rolled `solveWithL` / `luSolve` the
TypeScript uses, so a shared arithmetic bug would have to occur twice, in two
languages, in two different libraries.

Reference side: `numpy.linalg.solve` on the EXPLICITLY assembled saddle matrix
K = [[Abar, C^T], [C, 0]] — a dense LU of the whole (3n+k) system, sharing
nothing with the reformulation.

Right-hand sides (both shapes [DESIGN] 4.1 has to serve, and the same two the
committed CR rows use, so the numbers are directly comparable row for row):

  gradient   [b; 0]   b = splitmix32 uniforms on [-1,1), seed 1000
                      == CR's `gradient:rand0` row.  A physical
                      `gradientAnalytical` would be better still, but the oracle
                      has only a finite-difference dE (`tpe_stage1_oracle.py`
                      `finite_difference_dE`), which costs 6n O(n^2) energy
                      evaluations and would put N=480 out of reach.  The
                      reformulation's error does not depend on b being physical
                      — the ensemble rows in CR's own artifact are the evidence
                      for that — and the seeded draw is the HARDER case: it is
                      the row that sits closest to the bar at every N.
  projection [0; d]   d = -Phi(gamma_0 + delta), delta = 1e-3 * uniforms(3n, 7777)
                      == CR's `projection:production` GATED row, bit for bit.

Uses (imports, does not copy) the same oracle machinery as
`bench/gpu/k1-calibration.py` and `oracle/check_kappa_peredge.py`, and the same
trefoil parametrization as `src/core/fixtures.ts` `trefoil(n)`, so the twin
describes the milestone's own fixture rather than a proxy.

Run (repo root; no numpy/scipy in the repo's own toolchain by design):

    uv run --with numpy --with scipy python oracle/tpe_phase2a_reformulation.py

Pass condition: at every N and for BOTH rhs shapes the relative error is at the
kappa * u_f64 floor, i.e. at or under CR's own bar max(1e-12, 10*kappaK*u_f64)
with kappaK READ from the committed [K1CAL] artifact (never transcribed — the
same rule that binds CR).  Prints a PASS/FAIL per row and exits non-zero if any
row misses, so it is usable as a check and not only as a report.
CPU-only (numpy/LAPACK); no GPU is touched.

@see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md 4.1, 5 (CR), 6
@see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-gate-calibration.md F
@see test/gpu/cr.test.ts (the TypeScript side this localises against)
@see bench/gpu/k1-calibration.py (same oracle imports, same trefoil, [K1CAL] producer)
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import scipy.linalg as la

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tpe_constraints_oracle import (  # noqa: E402
    barycenter_block,
    evaluate_constraint_set,
    total_length_block,
)
from tpe_stage1_oracle import (  # noqa: E402
    assemble_inner_product,
    expand_vector_inner_product,
)

# The optimizer's DEFAULTS (`src/core/optimizer.ts`), which is what CR measures with.
ALPHA, BETA, EPS = 3.0, 6.0, 1e-10
U_F64 = 2.0**-53

# N=960 is deliberately absent: the twin exists to localise a defect in the
# ALGEBRA, which is N-independent, and a dense (2884 x 2884) LU reference plus a
# pure-python O(n^2) assembly at N=960 costs minutes for information N=480
# already carries.  CR itself covers 960.
FIXTURE_N = [60, 120, 240, 480]

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
K1CAL = os.path.join(REPO, "bench", "results", "2026-09-04-gpu-phase2a-k1-calibration.json")


def trefoil(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Same closed-curve parametrization as `src/core/fixtures.ts` `trefoil(n)`
    (verbatim transcription), so the twin's systems are the milestone's own
    fixture rather than a proxy.
    @see src/core/fixtures.ts
    """
    vertices, edges = [], []
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
    return np.asarray(vertices, float), np.asarray(edges, int)


def splitmix32(seed: int):
    """CR's ensemble PRNG, transcribed from `test/gpu/cr.test.ts`.

    This is the ONE thing the twin does copy, and it must be: the point of the
    seeds is that both sides draw the SAME vector, so a divergence is the
    algebra and not the input.  Every operation is exactly specified in both
    languages (32-bit wrap, xor, shifts, / 2**32), so the two agree bit for bit
    — which is exactly why CR uses splitmix32 rather than a Box-Muller normal
    (ECMA-262 21.3.2 leaves Math.log / Math.cos implementation-approximated).
    """
    a = seed & 0xFFFFFFFF

    def nxt() -> float:
        nonlocal a
        a = (a + 0x9E3779B9) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 16)) * 0x21F0AAAD) & 0xFFFFFFFF
        t = ((t ^ (t >> 15)) * 0x735A2D97) & 0xFFFFFFFF
        return ((t ^ (t >> 15)) & 0xFFFFFFFF) / 4294967296.0

    return nxt


def uniforms(count: int, seed: int) -> np.ndarray:
    """Uniform draws on [-1,1) — same generator, same consumption order as
    `uniforms` in `test/gpu/cr.test.ts`."""
    nxt = splitmix32(seed)
    return np.asarray([2.0 * nxt() - 1.0 for _ in range(count)], float)


def bar_for(n: int) -> float:
    """CR's bar, max(1e-12, 10 * kappa_2(K,N) * u_f64), with kappa READ from the
    committed [K1CAL] artifact.  N=60 has no [K1CAL] row (its fixture list is
    64/120/240/480/960 — 64, not 60) and takes the 1e-12 floor, exactly as CR does.
    @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-gate-calibration.md F
    """
    with open(K1CAL) as fh:
        rows = json.load(fh)["rows"]
    kappa = {int(r["n"]): float(r["kappaK"]) for r in rows}
    if n not in kappa:
        return 1e-12
    return max(1e-12, 10.0 * kappa[n] * U_F64)


def reformulated_solve(A: np.ndarray, C: np.ndarray, n: int):
    """[DESIGN] 4.1 steps 1-3, built from scratch on numpy/LAPACK primitives.

    step 1  sigma = trace(A)/n, A_sigma = A + (sigma/n) * 1 1^T -> SPD, Cholesky.
    step 2  Schur:  y = Ahat^-1 r_top, lam = S^-1 (C y - r_bot), xhat = y - Z lam,
            with Z = Ahat^-1 C^T and S = C Z.
    step 3  Woodbury back to K^-1:  z = zhat + W M^-1 (U^T zhat), U = [P; 0],
            P = I_3 (x) (1/sqrt(n)), W = Khat^-1 U, M = sigma^-1 I_3 - U^T W.

    Ahat = I_3 (x) A_sigma is COORDINATE-MAJOR (block b is coordinate b over all
    vertices, rows [b*n, (b+1)*n)) — the same convention as `flatten`, the core's
    assembly, and `expand_vector_inner_product`.  The interleaved spelling would
    make every row below fail.
    """
    m = 3 * n
    k = C.shape[0]
    one = np.ones(n)
    sigma = float(np.trace(A) / n)
    A_sigma = A + (sigma / n) * np.outer(one, one)
    cf = la.cho_factor(A_sigma, lower=True)

    def ahat_inv(v: np.ndarray) -> np.ndarray:
        """Three independent n-solves, one per coordinate block."""
        return np.concatenate([la.cho_solve(cf, v[b * n : (b + 1) * n]) for b in range(3)])

    Z = np.column_stack([ahat_inv(C[r]) for r in range(k)])  # m x k
    S = C @ Z  # k x k, SPD
    P = np.zeros((m, 3))
    for b in range(3):
        P[b * n : (b + 1) * n, b] = 1.0 / math.sqrt(n)
    W = np.zeros((m + k, 3))
    for b in range(3):
        y = ahat_inv(P[:, b])
        lam = np.linalg.solve(S, C @ y)
        W[:m, b] = y - Z @ lam
        W[m:, b] = lam
    M = np.eye(3) / sigma - P.T @ W[:m, :]

    def solve(rhs: np.ndarray) -> np.ndarray:
        y = ahat_inv(rhs[:m])
        lam = np.linalg.solve(S, C @ y - rhs[m:])
        zhat = np.concatenate([y - Z @ lam, lam])
        return zhat + W @ np.linalg.solve(M, P.T @ zhat[:m])

    return solve, sigma, M


def main() -> int:
    print(f"[DESIGN] 4.1 steps 1-3, numpy twin of src/gpu/saddle.ts.  u_f64 = {U_F64:.6e}")
    print(f"kappa_2(K) read from {os.path.relpath(K1CAL, REPO)}")
    print()
    header = f"{'N':>5} {'k':>3} {'rhs':>22} {'relErr':>12} {'bar':>12} {'slack':>9}  verdict"
    print(header)
    print("-" * len(header))
    failures = 0
    for n in FIXTURE_N:
        V, E = trefoil(n)
        _, _, A, _ = assemble_inner_product(V, E, ALPHA, BETA, EPS)
        A = np.asarray(A, float)
        # x0 / L0 exactly as `measure` in test/gpu/cr.test.ts builds them: the
        # ARITHMETIC vertex mean and the raw sum of edge lengths.  NOT
        # `length_weighted_barycenter` — that would be a different constraint
        # set and the projection rhs below would stop matching CR's row.
        x0 = V.mean(axis=0)
        L0 = float(sum(np.linalg.norm(V[int(b)] - V[int(a)]) for a, b in E))
        blocks = [barycenter_block(x0), total_length_block(L0)]
        _, C, _ = evaluate_constraint_set(blocks, V, E)
        C = np.asarray(C, float)
        m, k = 3 * n, C.shape[0]

        # Reference: a dense LU of the EXPLICIT saddle matrix, sharing nothing
        # with the reformulation above.
        K = np.zeros((m + k, m + k))
        K[:m, :m] = expand_vector_inner_product(A)
        K[:m, m:] = C.T
        K[m:, :m] = C
        solve, sigma, M = reformulated_solve(A, C, n)

        # projection rhs: Phi at the DISPLACED iterate, K and C frozen at gamma_0
        # — the shape the frozen projection solve passes (`src/core/sobolev/lineSearch.ts:267`).
        du = uniforms(3 * n, 7777)
        moved = V + 1e-3 * du.reshape(n, 3)
        phi_moved, _, _ = evaluate_constraint_set(blocks, moved, E)

        rhss = [
            ("gradient:rand0", np.concatenate([uniforms(m, 1000), np.zeros(k)])),
            ("projection:production", np.concatenate([np.zeros(m), -np.asarray(phi_moved, float)])),
        ]
        bar = bar_for(n)
        for tag, rhs in rhss:
            zref = np.linalg.solve(K, rhs)
            z = solve(rhs)
            rel = float(np.linalg.norm(z - zref) / np.linalg.norm(zref))
            ok = rel <= bar
            failures += 0 if ok else 1
            print(
                f"{n:>5} {k:>3} {tag:>22} {rel:>12.4e} {bar:>12.4e} "
                f"{bar / rel:>8.2f}x  {'PASS' if ok else 'FAIL'}"
            )
        print(
            f"{'':>5} {'':>3} {'diag(M)':>22} "
            f"{M[0][0]:.4e} {M[1][1]:.4e} {M[2][2]:.4e}   1/sigma = {1.0 / sigma:.4e}"
        )
    print()
    if failures:
        print(f"TWIN RED: {failures} row(s) miss the bar — the defect is in the ALGEBRA of")
        print("[DESIGN] 4.1, not in the TypeScript.  The SPEC is what has to change.")
        return 1
    print("TWIN GREEN: every row at the kappa*u_f64 floor. If test/gpu/cr.test.ts is red,")
    print("the defect is in src/gpu/saddle.ts, not in the algebra of [DESIGN] 4.1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
