#!/usr/bin/env python3
"""Property checklist runner for the Stage-1 oracle outputs.

Runs the machine-checkable properties from the research results doc
(local_files/2026-07-02-sobolev-gradient-rsrch-results.md §E) against the
golden JSON files produced by tpe_stage1_oracle.py. This validates the ORACLE
itself, before any TypeScript exists: if these fail, the formulas (not a
transcription) are wrong and no TS work should start.

Independence notes (load-bearing):
- The quadratic-form identity (checks 4a/4b) re-evaluates the defining double
  sums (D_I differences / edge averages) from the formulas in the results doc,
  written HERE from scratch — it does NOT call the oracle's assembly. Two code
  paths for the same definition.
- The barycenter value/preservation checks recompute Phi from raw geometry here.
- Scaling / orientation-invariance checks call the oracle's own functions at two
  different inputs; the property being tested is input-covariance of the
  formulas, so using the same implementation on both inputs is sound.

Usage: python check_properties.py <fixture.json> <golden.json>
Exit code 0 iff all checks pass.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tpe_stage1_oracle as oracle  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def ordered_disjoint_pairs(edges: np.ndarray) -> list[tuple[int, int]]:
    m = edges.shape[0]
    sets = [set(map(int, edges[i])) for i in range(m)]
    return [(i, j) for i in range(m) for j in range(m) if sets[i].isdisjoint(sets[j])]


# --- Independent transcription of the defining bilinear forms (results doc §A) ---

def direct_quadratic_forms(
    V: np.ndarray, E: np.ndarray, alpha: float, beta: float, eps: float,
    u: np.ndarray, v: np.ndarray,
) -> tuple[float, float]:
    """u^T B v and u^T B0 v evaluated straight from the defining double sums."""
    s = (beta - 1.0) / alpha
    sigma = s - 1.0
    dist_exp = 2.0 * sigma + 1.0

    def geom(I: int):
        i1, i2 = map(int, E[I])
        e = V[i2] - V[i1]
        ln = float(np.linalg.norm(e))
        ell = ln + eps
        T = e / ln if ln >= 1e-14 else np.zeros(3)
        return i1, i2, ell, T

    high = 0.0
    low = 0.0
    for I, J in ordered_disjoint_pairs(E):
        i1, i2, ell_i, Ti = geom(I)
        j1, j2, ell_j, Tj = geom(J)
        w = 0.0
        w0 = 0.0
        for a in (i1, i2):
            for b in (j1, j2):
                d = V[a] - V[b]
                dn = float(np.linalg.norm(d)) + eps
                inv = 1.0 / dn**dist_exp
                w += inv
                cn = float(np.linalg.norm(np.cross(Ti, d))) + eps
                w0 += (cn**2 / dn**4) * inv
        w *= 0.25 * ell_i * ell_j
        w0 *= 0.25 * ell_i * ell_j
        # D_I u = (u_i2 - u_i1)/ell * T
        DIu = (u[i2] - u[i1]) / ell_i * Ti
        DJu = (u[j2] - u[j1]) / ell_j * Tj
        DIv = (v[i2] - v[i1]) / ell_i * Ti
        DJv = (v[j2] - v[j1]) / ell_j * Tj
        high += w * float(np.dot(DIu - DJu, DIv - DJv))
        uI = 0.5 * (u[i1] + u[i2]); uJ = 0.5 * (u[j1] + u[j2])
        vI = 0.5 * (v[i1] + v[i2]); vJ = 0.5 * (v[j1] + v[j2])
        low += w0 * (uI - uJ) * (vI - vJ)
    return high, low


def barycenter_phi(V: np.ndarray, E: np.ndarray, x0: np.ndarray) -> np.ndarray:
    phi = np.zeros(3)
    for i1, i2 in E:
        p1, p2 = V[int(i1)], V[int(i2)]
        phi += float(np.linalg.norm(p2 - p1)) * (0.5 * (p1 + p2) - x0)
    return phi


def phi_scale(V: np.ndarray, E: np.ndarray, x0: np.ndarray) -> float:
    L = sum(float(np.linalg.norm(V[int(b)] - V[int(a)])) for a, b in E)
    R = max(float(np.linalg.norm(V[i] - x0)) for i in range(V.shape[0]))
    return max(1.0, L * max(1.0, R))


def main(fixture_path: str, golden_path: str) -> int:
    fx = json.load(open(fixture_path))
    g = json.load(open(golden_path))
    V = np.asarray(fx["vertices"], float)
    E = np.asarray(fx["edges"], int)
    alpha, beta, eps = float(fx["alpha"]), float(fx["beta"]), float(fx["epsilon"])
    n = V.shape[0]

    B = np.asarray(g["B"], float)
    B0 = np.asarray(g["B0"], float)
    A = np.asarray(g["A"], float)
    A3 = np.asarray(g["A3"], float)
    C = np.asarray(g["C_barycenter"], float)
    dE = np.asarray(g["dE_flat"], float)
    gt = np.asarray(g["g_tilde_flat"], float)
    lam = np.asarray(g["lambda"], float)
    x0 = np.asarray(g["x0_barycenter_target"], float)

    print(f"== {fx.get('name', fixture_path)} (|V|={n}, |E|={E.shape[0]}) ==")

    # 1. Symmetry
    for name, M in (("B", B), ("B0", B0)):
        r = np.linalg.norm(M - M.T) / max(1.0, np.linalg.norm(M))
        check(f"symmetry {name}", r <= 1e-14, f"rel asym {r:.2e}")

    # 2. PSD of A
    evals = np.linalg.eigvalsh(A)
    a2 = float(np.linalg.norm(A, 2))
    check("A PSD", evals[0] >= -1e-9 * a2, f"lambda_min={evals[0]:.3e}, ||A||2={a2:.3e}")

    # 3. Constant null space
    r = float(np.linalg.norm(A @ np.ones(n))) / max(1.0, a2 * math.sqrt(n))
    check("A annihilates constants", r <= 1e-10, f"rel {r:.2e}")

    # 4. Quadratic-form identity vs independent direct sums
    u = np.sin(1.0 + 0.7 * np.arange(n))
    v = np.cos(0.3 + 1.3 * np.arange(n))
    qB_direct, qB0_direct = direct_quadratic_forms(V, E, alpha, beta, eps, u, v)
    qB = float(u @ B @ v)
    qB0 = float(u @ B0 @ v)
    rB = abs(qB - qB_direct) / max(1e-300, abs(qB_direct))
    rB0 = abs(qB0 - qB0_direct) / max(1e-300, abs(qB0_direct))
    check("uBv == direct sum", rB <= 1e-12, f"rel {rB:.2e} (uBv={qB:.6g})")
    check("uB0v == direct sum", rB0 <= 1e-12, f"rel {rB0:.2e} (uB0v={qB0:.6g})")

    # 4c. A3 is blockdiag(A, A, A)
    blk = np.zeros_like(A3)
    for c in range(3):
        blk[c * n:(c + 1) * n, c * n:(c + 1) * n] = A
    check("A3 == blockdiag(A,A,A)", np.array_equal(A3, blk), "exact")

    # 6. Barycenter value at init
    phi = barycenter_phi(V, E, x0)
    sc = phi_scale(V, E, x0)
    check("Phi(init) ~ 0", float(np.linalg.norm(phi)) <= 1e-12 * sc,
          f"|Phi|={np.linalg.norm(phi):.2e}, scale={sc:.3g}")

    # 7. Barycenter Jacobian vs finite differences (independent Phi here)
    eta = 1e-6
    h = np.sin(0.11 + 0.37 * np.arange(3 * n))  # deterministic direction, block layout
    hV = np.column_stack([h[0:n], h[n:2 * n], h[2 * n:3 * n]])
    dphi_fd = (barycenter_phi(V + eta * hV, E, x0) - phi) / eta
    dphi_C = C @ h
    rj = float(np.linalg.norm(dphi_fd - dphi_C)) / max(1.0, float(np.linalg.norm(dphi_fd)))
    check("C == dPhi (FD)", rj <= 1e-5, f"rel {rj:.2e}")

    # 8. Saddle residuals (recomputed from golden pieces)
    res_top = float(np.linalg.norm(A3 @ gt + C.T @ lam - dE)) / max(1.0, float(np.linalg.norm(dE)))
    res_bot = float(np.linalg.norm(C @ gt)) / max(1.0, float(np.linalg.norm(gt)))
    check("saddle top residual", res_top <= 1e-10, f"rel {res_top:.2e}")
    check("C g_tilde ~ 0", res_bot <= 1e-10, f"rel {res_bot:.2e}")
    check("oracle-reported residual", float(g["saddle_relative_residual"]) <= 1e-10,
          f"{g['saddle_relative_residual']:.2e}")

    # 9. Descent positivity
    slope = float(dE @ gt)
    check("dE . g_tilde > 0", slope > 0, f"{slope:.6g}")

    # 11/12. Accepted step: Armijo + barycenter preservation
    ls = g["line_search_step"]
    check("line search accepted", bool(ls["accepted"]), f"tau={ls.get('tau')}")
    if ls["accepted"]:
        e0, e1, tau, m = ls["energy_before"], ls["energy_after"], ls["tau"], ls["slope"]
        check("energy decreases", e1 < e0, f"{e0:.6g} -> {e1:.6g}")
        check("Armijo", e1 <= e0 - 1e-4 * tau * m, f"E1={e1:.6g} bound={e0 - 1e-4 * tau * m:.6g}")
        Vnew = np.asarray(ls["vertices"], float)
        phi_new = barycenter_phi(Vnew, E, x0)
        sc_new = phi_scale(Vnew, E, x0)
        check("barycenter preserved", float(np.linalg.norm(phi_new)) <= 1e-10 * sc_new,
              f"|Phi(new)|={np.linalg.norm(phi_new):.2e}")

    # 13. Scaling laws (oracle called at two inputs; eps negligible at these scales)
    c = 2.0
    E_1 = oracle.energy(V, E, alpha, beta, eps)
    E_c = oracle.energy(c * V, E, alpha, beta, eps)
    expo_E = alpha - beta + 2.0  # -1 for (3,6)
    rE = abs(E_c - c**expo_E * E_1) / abs(E_1)
    check(f"E(c*g) = c^{expo_E:g} E(g)", rE <= 1e-8, f"rel {rE:.2e}")
    _, _, A_c, info = oracle.assemble_inner_product(c * V, E, alpha, beta, eps)
    expo_A = -info["two_sigma_plus_one"]  # -7/3 for (3,6)
    rA = float(np.linalg.norm(A_c - c**expo_A * A)) / float(np.linalg.norm(A))
    check(f"A(c*g) = c^{expo_A:.4g} A(g)", rA <= 1e-8, f"rel {rA:.2e}")

    # 5(E). Orientation invariance: flip edge 0
    E_flip = E.copy()
    E_flip[0] = E_flip[0][::-1]
    e_flip = oracle.energy(V, E_flip, alpha, beta, eps)
    rEf = abs(e_flip - g["energy"]) / abs(g["energy"])
    check("energy orientation-invariant", rEf <= 1e-12, f"rel {rEf:.2e}")
    Bf, B0f, Af, _ = oracle.assemble_inner_product(V, E_flip, alpha, beta, eps)
    rBf = float(np.linalg.norm(Af - A)) / max(1.0, float(np.linalg.norm(A)))
    check("A orientation-invariant", rBf <= 1e-12, f"rel {rBf:.2e}")
    _, Cf = oracle.barycenter_phi_and_C(V, E_flip, x0)
    rCf = float(np.linalg.norm(Cf - C)) / max(1.0, float(np.linalg.norm(C)))
    check("C orientation-invariant", rCf <= 1e-12, f"rel {rCf:.2e}")

    print()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: check_properties.py fixture.json golden.json", file=sys.stderr)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")
    raise SystemExit(0)
