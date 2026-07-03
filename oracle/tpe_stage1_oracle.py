#!/usr/bin/env python3
"""
Stage 1 reference oracle for the discrete tangent-point energy and fractional
Sobolev gradient used by the browser app.

Usage:
    python tpe_stage1_oracle.py input.json output.json

Input JSON:
{
  "vertices": [[x,y,z], ...],
  "edges": [[i,j], ...],
  "alpha": 3,
  "beta": 6,
  "epsilon": 1e-10,
  "finite_difference_h": 1e-6   # optional
}

The implementation intentionally uses dense matrices and direct symmetric
indefinite solves. It is deterministic and depends only on numpy/scipy.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

DEGENERATE_TOL = 1.0e-14
DEFAULT_FD_H = 1.0e-6


@dataclass
class EdgeGeom:
    i1: int
    i2: int
    e: np.ndarray
    ell_raw: float
    ell_metric: float
    tangent: np.ndarray
    midpoint: np.ndarray


def as_vertices(vertices: Any) -> np.ndarray:
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must have shape (n, 3)")
    return v


def as_edges(edges: Any, n: int) -> np.ndarray:
    e = np.asarray(edges, dtype=int)
    if e.ndim != 2 or e.shape[1] != 2:
        raise ValueError("edges must have shape (m, 2)")
    if np.any(e < 0) or np.any(e >= n):
        raise ValueError("edge index out of bounds")
    return e


def safe_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def safe_unit(x: np.ndarray) -> np.ndarray:
    n = safe_norm(x)
    if n < DEGENERATE_TOL:
        return np.zeros(3, dtype=float)
    return x / n


def flatten_vec3_block(vecs: np.ndarray) -> np.ndarray:
    """Vec3 per vertex -> [x_0..x_n-1, y_0.., z_0..]."""
    a = np.asarray(vecs, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("expected (n, 3) Vec3 array")
    return np.concatenate([a[:, 0], a[:, 1], a[:, 2]])


def unflatten_vec3_block(flat: np.ndarray) -> np.ndarray:
    f = np.asarray(flat, dtype=float).reshape(-1)
    if f.size % 3 != 0:
        raise ValueError("flat vector length must be divisible by 3")
    n = f.size // 3
    return np.column_stack([f[0:n], f[n:2*n], f[2*n:3*n]])


def block_index(coord: int, vertex: int, n: int) -> int:
    return coord * n + vertex


def compute_disjoint_pairs(edges: np.ndarray) -> List[List[int]]:
    m = edges.shape[0]
    pairs: List[List[int]] = [[] for _ in range(m)]
    sets = [set(map(int, edges[i])) for i in range(m)]
    for i in range(m):
        for j in range(m):
            if sets[i].isdisjoint(sets[j]):
                pairs[i].append(j)
    return pairs


def edge_geometries(vertices: np.ndarray, edges: np.ndarray, eps: float) -> List[EdgeGeom]:
    geoms: List[EdgeGeom] = []
    for i1, i2 in edges:
        p1 = vertices[int(i1)]
        p2 = vertices[int(i2)]
        ev = p2 - p1
        ell_raw = safe_norm(ev)
        ell_metric = ell_raw + eps
        tangent = safe_unit(ev)
        midpoint = 0.5 * (p1 + p2)
        geoms.append(EdgeGeom(int(i1), int(i2), ev, ell_raw, ell_metric, tangent, midpoint))
    return geoms


def energy(vertices: np.ndarray, edges: np.ndarray, alpha: float, beta: float, eps: float) -> float:
    """App energy exactly: ordered disjoint pairs, then divide by 2."""
    disjoint = compute_disjoint_pairs(edges)
    total = 0.0
    for I, Js in enumerate(disjoint):
        i1, i2 = map(int, edges[I])
        eI = vertices[i2] - vertices[i1]
        ell_I = safe_norm(eI) + eps
        for J in Js:
            j1, j2 = map(int, edges[J])
            eJ = vertices[j2] - vertices[j1]
            ell_J = safe_norm(eJ) + eps
            sum_k = 0.0
            for i in (i1, i2):
                for j in (j1, j2):
                    d = vertices[i] - vertices[j]
                    d_norm = safe_norm(d) + eps
                    c = np.cross(eI, d)
                    c_norm = safe_norm(c) + eps
                    sum_k += (c_norm ** alpha) / (d_norm ** beta)
            total += 0.25 * (ell_I ** (1.0 - alpha)) * ell_J * sum_k
    return 0.5 * total


def finite_difference_dE(
    vertices: np.ndarray,
    edges: np.ndarray,
    alpha: float,
    beta: float,
    eps: float,
    h: float = DEFAULT_FD_H,
) -> np.ndarray:
    n = vertices.shape[0]
    grad = np.zeros((n, 3), dtype=float)
    e0 = energy(vertices, edges, alpha, beta, eps)
    for v in range(n):
        for c in range(3):
            perturbed = vertices.copy()
            perturbed[v, c] += h
            e1 = energy(perturbed, edges, alpha, beta, eps)
            grad[v, c] = (e1 - e0) / h
    return grad


def tangent_point_kernel(p: np.ndarray, q: np.ndarray, T: np.ndarray, alpha: float, beta: float, eps: float) -> float:
    d = p - q
    d_norm = safe_norm(d) + eps
    c_norm = safe_norm(np.cross(T, d)) + eps
    return (c_norm ** alpha) / (d_norm ** beta)


def assemble_inner_product(
    vertices: np.ndarray,
    edges: np.ndarray,
    alpha: float,
    beta: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Return B_high, B_low, A = B_high + B_low as n-by-n matrices."""
    n = vertices.shape[0]
    geoms = edge_geometries(vertices, edges, eps)
    disjoint = compute_disjoint_pairs(edges)

    s = (beta - 1.0) / alpha
    sigma = s - 1.0
    dist_exp = 2.0 * sigma + 1.0

    B = np.zeros((n, n), dtype=float)
    B0 = np.zeros((n, n), dtype=float)

    for I, Js in enumerate(disjoint):
        gi = geoms[I]
        idx_i = (gi.i1, gi.i2)
        ell_i = gi.ell_metric
        Ti = gi.tangent
        for J in Js:
            gj = geoms[J]
            idx_j = (gj.i1, gj.i2)
            ell_j = gj.ell_metric
            Tj = gj.tangent

            # High-order weight w_IJ.
            sum_dist = 0.0
            # Low-order weight w^low_IJ.
            sum_low = 0.0
            for ia in idx_i:
                for jb in idx_j:
                    d = vertices[ia] - vertices[jb]
                    d_norm = safe_norm(d) + eps
                    inv_frac = 1.0 / (d_norm ** dist_exp)
                    sum_dist += inv_frac
                    k24 = tangent_point_kernel(vertices[ia], vertices[jb], Ti, 2.0, 4.0, eps)
                    sum_low += k24 * inv_frac

            w = 0.25 * ell_i * ell_j * sum_dist
            w_low = 0.25 * ell_i * ell_j * sum_low
            dot_t = float(np.dot(Ti, Tj))

            for a in range(2):
                for b in range(2):
                    sign = 1.0 if ((a + b) % 2 == 0) else -1.0
                    ia = idx_i[a]
                    ib = idx_i[b]
                    ja = idx_j[a]
                    jb = idx_j[b]

                    B[ia, ib] += sign * w / (ell_i * ell_i)
                    B[ia, jb] -= sign * w * dot_t / (ell_i * ell_j)
                    B[ja, jb] += sign * w / (ell_j * ell_j)
                    B[ja, ib] -= sign * w * dot_t / (ell_i * ell_j)

                    q = 0.25 * w_low
                    B0[ia, ib] += q
                    B0[ia, jb] -= q
                    B0[ja, ib] -= q
                    B0[ja, jb] += q

    # Remove roundoff-level asymmetry from accumulation order.
    B = 0.5 * (B + B.T)
    B0 = 0.5 * (B0 + B0.T)
    A = B + B0
    info = {"s": s, "sigma": sigma, "two_sigma_plus_one": dist_exp}
    return B, B0, A, info


def expand_vector_inner_product(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    z = np.zeros_like(A)
    return np.block([[A, z, z], [z, A, z], [z, z, A]])


def length_weighted_barycenter(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    total = 0.0
    accum = np.zeros(3, dtype=float)
    for i1, i2 in edges:
        p1 = vertices[int(i1)]
        p2 = vertices[int(i2)]
        ell = safe_norm(p2 - p1)
        midpoint = 0.5 * (p1 + p2)
        total += ell
        accum += ell * midpoint
    if total < DEGENERATE_TOL:
        # Last-resort fallback for a completely degenerate graph.
        return np.mean(vertices, axis=0)
    return accum / total


def barycenter_phi_and_C(
    vertices: np.ndarray,
    edges: np.ndarray,
    x0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Phi = sum_I ell_I (midpoint_I - x0), C = dPhi/dgamma.

    C uses block coordinate flattening: [x vertices, y vertices, z vertices].
    """
    n = vertices.shape[0]
    phi = np.zeros(3, dtype=float)
    C = np.zeros((3, 3 * n), dtype=float)

    for i1, i2 in edges:
        i1 = int(i1)
        i2 = int(i2)
        p1 = vertices[i1]
        p2 = vertices[i2]
        ev = p2 - p1
        ell = safe_norm(ev)
        T = safe_unit(ev)
        midpoint = 0.5 * (p1 + p2)
        r = midpoint - x0
        phi += ell * r

        # d[ell * r_k] = d ell * r_k + ell * d midpoint_k
        # d ell / d p1_c = -T_c, d ell / d p2_c = +T_c.
        for out_coord in range(3):
            for in_coord in range(3):
                delta = 1.0 if out_coord == in_coord else 0.0
                C[out_coord, block_index(in_coord, i1, n)] += -T[in_coord] * r[out_coord] + 0.5 * ell * delta
                C[out_coord, block_index(in_coord, i2, n)] += +T[in_coord] * r[out_coord] + 0.5 * ell * delta
    return phi, C


def barycenter_scale(vertices: np.ndarray, edges: np.ndarray, x0: np.ndarray) -> float:
    L = 0.0
    R = 0.0
    for i1, i2 in edges:
        p1 = vertices[int(i1)]
        p2 = vertices[int(i2)]
        L += safe_norm(p2 - p1)
        R = max(R, safe_norm(p1 - x0), safe_norm(p2 - x0))
    return max(1.0, L * max(1.0, R))


def solve_saddle(A3: np.ndarray, C: np.ndarray, rhs_top: np.ndarray, rhs_bottom: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    k = C.shape[0]
    if rhs_bottom is None:
        rhs_bottom = np.zeros(k, dtype=float)
    K = np.block([
        [A3, C.T],
        [C, np.zeros((k, k), dtype=float)],
    ])
    rhs = np.concatenate([rhs_top, rhs_bottom])
    try:
        sol = la.solve(K, rhs, assume_a="sym", check_finite=True)
    except Exception:
        # Fallback for rank-deficient pathological cases. The residual is reported.
        sol, *_ = la.lstsq(K, rhs, check_finite=True)
    residual = safe_norm(K @ sol - rhs) / max(1.0, safe_norm(rhs))
    return sol[: A3.shape[0]], sol[A3.shape[0] :], residual


def solve_constrained_gradient(
    vertices: np.ndarray,
    edges: np.ndarray,
    alpha: float,
    beta: float,
    eps: float,
    dE_vec3: np.ndarray,
    x0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, Dict[str, float]]:
    B, B0, A, info = assemble_inner_product(vertices, edges, alpha, beta, eps)
    A3 = expand_vector_inner_product(A)
    _phi, C = barycenter_phi_and_C(vertices, edges, x0)
    rhs = flatten_vec3_block(dE_vec3)
    g_flat, lam, residual = solve_saddle(A3, C, rhs)
    return unflatten_vec3_block(g_flat), lam, B, B0, A, residual, info


def l2_curve_norm_vec3(values: np.ndarray, vertices: np.ndarray, edges: np.ndarray) -> float:
    """Mass-lumped discrete curve L2 norm: sum_I ell_I/2*(|v_i1|^2+|v_i2|^2)."""
    total = 0.0
    for i1, i2 in edges:
        i1 = int(i1)
        i2 = int(i2)
        ell = safe_norm(vertices[i2] - vertices[i1])
        total += 0.5 * ell * (float(np.dot(values[i1], values[i1])) + float(np.dot(values[i2], values[i2])))
    return math.sqrt(max(0.0, total))


def project_barycenter(
    vertices: np.ndarray,
    edges: np.ndarray,
    alpha: float,
    beta: float,
    eps: float,
    x0: np.ndarray,
    tol_abs: float = 1.0e-10,
    tol_rel: float = 1.0e-10,
    max_iter: int = 8,
) -> Tuple[np.ndarray, bool, int, float]:
    cur = vertices.copy()
    final_phi_norm = math.inf
    for it in range(max_iter + 1):
        phi, C = barycenter_phi_and_C(cur, edges, x0)
        scale = barycenter_scale(cur, edges, x0)
        final_phi_norm = safe_norm(phi)
        if final_phi_norm <= max(tol_abs, tol_rel * scale):
            return cur, True, it, final_phi_norm
        if it == max_iter:
            break
        try:
            _B, _B0, A, _info = assemble_inner_product(cur, edges, alpha, beta, eps)
            A3 = expand_vector_inner_product(A)
            x_flat, _mu, _res = solve_saddle(A3, C, np.zeros(3 * cur.shape[0], dtype=float), -phi)
            step = unflatten_vec3_block(x_flat)
            if not np.all(np.isfinite(step)):
                return cur, False, it, final_phi_norm
            cur = cur + step
        except Exception:
            return cur, False, it, final_phi_norm
    return cur, False, max_iter, final_phi_norm


def line_search_step(
    vertices: np.ndarray,
    edges: np.ndarray,
    alpha: float,
    beta: float,
    eps: float,
    dE_vec3: np.ndarray,
    g_tilde: np.ndarray,
    x0: np.ndarray,
    armijo_c1: float = 1.0e-4,
    shrink: float = 0.5,
    tau0: float = 1.0,
    tau_min: float = 1.0e-12,
) -> Dict[str, Any]:
    e0 = energy(vertices, edges, alpha, beta, eps)
    grad_norm = l2_curve_norm_vec3(g_tilde, vertices, edges)
    if not np.isfinite(grad_norm) or grad_norm <= 0.0:
        return {
            "accepted": False,
            "reason": "zero_or_nonfinite_gradient_norm",
            "tau": 0.0,
            "energy_before": e0,
            "energy_after": e0,
            "vertices": vertices.tolist(),
            "projection_iterations": 0,
            "projection_phi_norm": None,
        }

    direction = g_tilde / grad_norm
    slope = float(np.dot(flatten_vec3_block(dE_vec3), flatten_vec3_block(direction)))
    if not np.isfinite(slope) or slope <= 0.0:
        return {
            "accepted": False,
            "reason": "not_a_descent_direction",
            "tau": 0.0,
            "energy_before": e0,
            "energy_after": e0,
            "slope": slope,
            "vertices": vertices.tolist(),
            "projection_iterations": 0,
            "projection_phi_norm": None,
        }

    tau = tau0
    last_reason = "tau_below_min"
    while tau >= tau_min:
        raw = vertices - tau * direction
        projected, ok, proj_iters, phi_norm = project_barycenter(raw, edges, alpha, beta, eps, x0)
        if ok and np.all(np.isfinite(projected)):
            e1 = energy(projected, edges, alpha, beta, eps)
            if np.isfinite(e1) and e1 <= e0 - armijo_c1 * tau * slope:
                return {
                    "accepted": True,
                    "tau": tau,
                    "energy_before": e0,
                    "energy_after": e1,
                    "slope": slope,
                    "gradient_l2_norm": grad_norm,
                    "vertices": projected.tolist(),
                    "projection_iterations": proj_iters,
                    "projection_phi_norm": phi_norm,
                    "direction": direction.tolist(),
                }
            last_reason = "armijo_failed"
        else:
            last_reason = "projection_failed"
        tau *= shrink

    return {
        "accepted": False,
        "reason": last_reason,
        "tau": 0.0,
        "energy_before": e0,
        "energy_after": e0,
        "slope": slope,
        "gradient_l2_norm": grad_norm,
        "vertices": vertices.tolist(),
        "projection_iterations": None,
        "projection_phi_norm": None,
    }


def to_jsonable_matrix(M: np.ndarray) -> List[List[float]]:
    return [[float(x) for x in row] for row in np.asarray(M, dtype=float)]


def to_jsonable_vec3(V: np.ndarray) -> List[List[float]]:
    return [[float(x), float(y), float(z)] for x, y, z in np.asarray(V, dtype=float)]


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print("Usage: python tpe_stage1_oracle.py input.json output.json", file=sys.stderr)
        return 2
    with open(argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    vertices = as_vertices(data["vertices"])
    edges = as_edges(data["edges"], vertices.shape[0])
    alpha = float(data.get("alpha", 3.0))
    beta = float(data.get("beta", 6.0))
    eps = float(data.get("epsilon", 1.0e-10))
    h = float(data.get("finite_difference_h", DEFAULT_FD_H))

    E = energy(vertices, edges, alpha, beta, eps)
    dE = finite_difference_dE(vertices, edges, alpha, beta, eps, h)
    x0 = length_weighted_barycenter(vertices, edges)
    g_tilde, lambdas, B, B0, A, saddle_residual, order_info = solve_constrained_gradient(
        vertices, edges, alpha, beta, eps, dE, x0
    )
    A3 = expand_vector_inner_product(A)
    phi, C = barycenter_phi_and_C(vertices, edges, x0)
    step = line_search_step(vertices, edges, alpha, beta, eps, dE, g_tilde, x0)

    out: Dict[str, Any] = {
        "conventions": {
            "flattening": "coordinate-block: [x0..xN-1, y0..yN-1, z0..zN-1]",
            "ordered_disjoint_pairs": True,
            "energy_has_extra_one_half": True,
            "inner_product_pair_factor": 1.0,
            "epsilon_after_norms_in_metric_weights": True,
            "degenerate_unit_tangent_tol": DEGENERATE_TOL,
        },
        "alpha": alpha,
        "beta": beta,
        "epsilon": eps,
        "finite_difference_h": h,
        "orders": order_info,
        "energy": float(E),
        "dE": to_jsonable_vec3(dE),
        "dE_flat": [float(x) for x in flatten_vec3_block(dE)],
        "B": to_jsonable_matrix(B),
        "B0": to_jsonable_matrix(B0),
        "A": to_jsonable_matrix(A),
        "A3": to_jsonable_matrix(A3),
        "x0_barycenter_target": [float(x) for x in x0],
        "Phi_barycenter": [float(x) for x in phi],
        "C_barycenter": to_jsonable_matrix(C),
        "g_tilde": to_jsonable_vec3(g_tilde),
        "g_tilde_flat": [float(x) for x in flatten_vec3_block(g_tilde)],
        "lambda": [float(x) for x in lambdas],
        "saddle_relative_residual": float(saddle_residual),
        "gradient_l2_norm": float(l2_curve_norm_vec3(g_tilde, vertices, edges)),
        "line_search_step": step,
    }

    with open(argv[2], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
