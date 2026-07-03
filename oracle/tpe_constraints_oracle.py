#!/usr/bin/env python3
"""
Constraints oracle for the Sobolev flow constraint milestones (M1: total
length; M2: per-edge lengths + point pins).

Imports the stage-1 oracle (`tpe_stage1_oracle.py`, read-only deliverable)
and adds the generalized ConstraintSet machinery mirroring the TS side
(`src/core/sobolev/constraintSet.ts`): stacked constraint rows
(barycenter FIRST, spec §3.2), a per-block projection stopping tolerance,
and the set-generic line-search step. Emits `<fixture>-<mode>.json` goldens
and runs embedded property checks (exit code 0 iff all pass).

Spec: docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md
(§2 rows/signs, §3.1 stacking, §3.3 per-block tolerance — OUR invention,
not paper-sourced, see local_files/2026-07-02-sobolev-formula-audit.md item 9).

Usage:
    python tpe_constraints_oracle.py fixture.json golden.json [mode]

mode: "length" (default, M1) = [barycenter, totalLength];
      "edgelengths" (M2)     = [barycenter, edgeLengths];
      "point" (M2)           = [barycenter, point(vertex 0, initial position)].
"""
from __future__ import annotations

import json
import math
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from tpe_stage1_oracle import (
    DEFAULT_FD_H,
    as_edges,
    as_vertices,
    assemble_inner_product,
    barycenter_phi_and_C,
    barycenter_scale,
    block_index,
    energy,
    expand_vector_inner_product,
    finite_difference_dE,
    flatten_vec3_block,
    l2_curve_norm_vec3,
    length_weighted_barycenter,
    safe_norm,
    safe_unit,
    solve_saddle,
    to_jsonable_vec3,
    unflatten_vec3_block,
)

# A block mirrors the TS ConstraintBlock: kind + evaluate -> (phi, C) + scale.
Block = Dict[str, Any]


def total_length(vertices: np.ndarray, edges: np.ndarray) -> float:
    """Raw total length L = sum_I ||e_I||, NO +eps (constraints are geometric).

    Same raw-length rule as barycenter_phi_and_C / the TS totalLength.
    """
    L = 0.0
    for i1, i2 in edges:
        L += safe_norm(vertices[int(i2)] - vertices[int(i1)])
    return L


def total_length_phi_and_C(
    vertices: np.ndarray, edges: np.ndarray, L0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Phi = [L0 - sum_I ell_I] (paper sign: target minus current), 1 row.

    Row: +T_I at i1's columns, -T_I at i2's (dPhi = -sum dell_I with
    dell_I = T_I . (dgamma_i2 - dgamma_i1)); coordinate-block columns via
    block_index. Degenerate edges contribute T = 0 (safe_unit's 1e-14 guard,
    same constant as the TS side). Spec §2.
    """
    n = vertices.shape[0]
    row = np.zeros(3 * n, dtype=float)
    L = 0.0
    for i1, i2 in edges:
        i1 = int(i1)
        i2 = int(i2)
        ev = vertices[i2] - vertices[i1]
        L += safe_norm(ev)
        T = safe_unit(ev)
        for c in range(3):
            row[block_index(c, i1, n)] += T[c]
            row[block_index(c, i2, n)] += -T[c]
    return np.array([L0 - L], dtype=float), row.reshape(1, 3 * n)


def barycenter_block(x0: np.ndarray) -> Block:
    return {
        "kind": "barycenter",
        "evaluate": lambda V, E: barycenter_phi_and_C(V, E, x0),
        "scale": lambda V, E: barycenter_scale(V, E, x0),
    }


def total_length_block(L0: float) -> Block:
    # scale max(1, L) is OUR tunable choice, not paper-sourced (spec §3.3;
    # formula-audit item 9).
    return {
        "kind": "totalLength",
        "evaluate": lambda V, E: total_length_phi_and_C(V, E, L0),
        "scale": lambda V, E: max(1.0, total_length(V, E)),
    }


def edge_lengths(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Raw per-edge lengths ell_I = ||e_I|| in edge order, NO +eps.

    Constraints are geometric — same raw-length rule as total_length /
    barycenter_phi_and_C (spec §2)."""
    return np.array(
        [safe_norm(vertices[int(i2)] - vertices[int(i1)]) for i1, i2 in edges],
        dtype=float,
    )


def edge_lengths_phi_and_C(
    vertices: np.ndarray, edges: np.ndarray, ell0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Phi_I = ell0_I - ell_I (paper sign, target minus current), |E| rows.

    Row I: +T_I at i1's columns, -T_I at i2's, THAT EDGE ONLY (spec §2:
    dPhi_I = -dell_I, dell_I = T_I . (dgamma_i2 - dgamma_i1)). The totalLength
    row is exactly the SUM of these rows — the §3.4 rank rule. Degenerate
    edges contribute T = 0 (safe_unit's 1e-14 guard)."""
    n = vertices.shape[0]
    m = edges.shape[0]
    phi = np.zeros(m, dtype=float)
    C = np.zeros((m, 3 * n), dtype=float)
    for r, (i1, i2) in enumerate(edges):
        i1 = int(i1)
        i2 = int(i2)
        ev = vertices[i2] - vertices[i1]
        phi[r] = float(ell0[r]) - safe_norm(ev)
        T = safe_unit(ev)
        for c in range(3):
            C[r, block_index(c, i1, n)] += T[c]
            C[r, block_index(c, i2, n)] += -T[c]
    return phi, C


def point_phi_and_C(
    vertices: np.ndarray, vertex_index: int, target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Phi = gamma_i - x_i (paper-verbatim sign: CURRENT minus target, spec §2),
    3 rows; C is the identity block C[r, block_index(r, i, n)] = 1 — no
    length terms."""
    n = vertices.shape[0]
    C = np.zeros((3, 3 * n), dtype=float)
    for r in range(3):
        C[r, block_index(r, vertex_index, n)] = 1.0
    return vertices[vertex_index] - target, C


def edge_lengths_block(ell0: np.ndarray) -> Block:
    # scale max(1, L) is OUR tunable choice, not paper-sourced (spec §3.3;
    # formula-audit item 9).
    return {
        "kind": "edgeLengths",
        "evaluate": lambda V, E: edge_lengths_phi_and_C(V, E, ell0),
        "scale": lambda V, E: max(1.0, total_length(V, E)),
    }


def point_block(vertex_index: int, target: np.ndarray) -> Block:
    # scale max(1, R), R = max distance from ANY vertex to the pin target —
    # OUR tunable choice, not paper-sourced (spec §3.3).
    t = np.asarray(target, dtype=float)
    return {
        "kind": "point",
        "evaluate": lambda V, E: point_phi_and_C(V, vertex_index, t),
        "scale": lambda V, E: max(1.0, float(max(safe_norm(v - t) for v in V))),
    }


def evaluate_constraint_set(
    blocks: List[Block], vertices: np.ndarray, edges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Stack every block's phi/C rows in list order (barycenter first when
    present, spec §3.2). Returns (phi, C, row_counts per block)."""
    phis: List[np.ndarray] = []
    rows: List[np.ndarray] = []
    counts: List[int] = []
    for b in blocks:
        phi_b, C_b = b["evaluate"](vertices, edges)
        phis.append(np.asarray(phi_b, dtype=float).reshape(-1))
        rows.append(np.asarray(C_b, dtype=float))
        counts.append(phis[-1].size)
    n3 = 3 * vertices.shape[0]
    if not blocks:
        return np.zeros(0), np.zeros((0, n3)), []
    return np.concatenate(phis), np.vstack(rows), counts


def solve_constrained_gradient_set(
    vertices: np.ndarray,
    edges: np.ndarray,
    alpha: float,
    beta: float,
    eps: float,
    dE_vec3: np.ndarray,
    blocks: List[Block],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Saddle solve [[A3, C^T], [C, 0]][g; lam] = [dE; 0] with stacked C."""
    _B, _B0, A, _info = assemble_inner_product(vertices, edges, alpha, beta, eps)
    A3 = expand_vector_inner_product(A)
    _phi, C, _counts = evaluate_constraint_set(blocks, vertices, edges)
    g_flat, lam, residual = solve_saddle(A3, C, flatten_vec3_block(dE_vec3))
    return unflatten_vec3_block(g_flat), lam, residual


def constraint_set_converged(
    blocks: List[Block],
    vertices: np.ndarray,
    edges: np.ndarray,
    phi: np.ndarray,
    counts: List[int],
    tol_abs: float,
    tol_rel: float,
) -> bool:
    """Per-block stopping rule (spec §3.3, OUR invention): converged iff EVERY
    block's phi slice satisfies ||phi_b||_2 <= max(tol_abs, tol_rel*scale_b).
    Reduces exactly to the stage-1 barycenter rule for the barycenter-only set."""
    offset = 0
    for b, k in zip(blocks, counts):
        phi_b = phi[offset : offset + k]
        offset += k
        if safe_norm(phi_b) > max(tol_abs, tol_rel * b["scale"](vertices, edges)):
            return False
    return True


def project_constraint_set(
    vertices: np.ndarray,
    edges: np.ndarray,
    alpha: float,
    beta: float,
    eps: float,
    blocks: List[Block],
    tol_abs: float = 1.0e-10,
    tol_rel: float = 1.0e-10,
    max_iter: int = 8,
) -> Tuple[np.ndarray, bool, int, float]:
    """Set-generic mirror of the stage-1 project_barycenter: same
    check-before-correct loop (max_iter + 1 checks), same reassembly of A per
    iterate, same failure semantics; only the convergence test is per-block and
    the reported phi_norm is the STACKED ||Phi||_2."""
    cur = vertices.copy()
    final_phi_norm = math.inf
    for it in range(max_iter + 1):
        phi, C, counts = evaluate_constraint_set(blocks, cur, edges)
        final_phi_norm = safe_norm(phi)
        if constraint_set_converged(blocks, cur, edges, phi, counts, tol_abs, tol_rel):
            return cur, True, it, final_phi_norm
        if it == max_iter:
            break
        try:
            _B, _B0, A, _info = assemble_inner_product(cur, edges, alpha, beta, eps)
            A3 = expand_vector_inner_product(A)
            x_flat, _mu, _res = solve_saddle(
                A3, C, np.zeros(3 * cur.shape[0], dtype=float), -phi
            )
            step = unflatten_vec3_block(x_flat)
            if not np.all(np.isfinite(step)):
                return cur, False, it, final_phi_norm
            cur = cur + step
        except Exception:
            return cur, False, it, final_phi_norm
    return cur, False, max_iter, final_phi_norm


def line_search_step_set(
    vertices: np.ndarray,
    edges: np.ndarray,
    alpha: float,
    beta: float,
    eps: float,
    dE_vec3: np.ndarray,
    g_tilde: np.ndarray,
    blocks: List[Block],
    armijo_c1: float = 1.0e-4,
    shrink: float = 0.5,
    tau0: float = 1.0,
    tau_min: float = 1.0e-12,
) -> Dict[str, Any]:
    """Set-generic mirror of the stage-1 line_search_step (identical control
    flow and payloads; projection swapped for project_constraint_set)."""
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
        projected, ok, proj_iters, phi_norm = project_constraint_set(
            raw, edges, alpha, beta, eps, blocks
        )
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


# ---------------------------------------------------------------------------
# Embedded property checks (pattern of check_properties.py): exit 0 iff all
# pass, so this script is both golden generator and its own property gate.
# ---------------------------------------------------------------------------

_FAILURES: List[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not ok:
        _FAILURES.append(name)


def fd_jacobian_check(
    blocks: List[Block], vertices: np.ndarray, edges: np.ndarray, eta: float = 1.0e-6
) -> Tuple[float, bool]:
    """Central-difference FD check of the stacked Jacobian (spec §4.4.1 /
    rsrch-results §E prop 7 pattern): deterministic direction, rel gate 1e-6."""
    n3 = 3 * vertices.shape[0]
    # Deterministic, non-trivial direction (no RNG — oracle must be reproducible).
    h = np.array([math.sin(1.0 + 0.7 * i) for i in range(n3)], dtype=float)
    h /= safe_norm(h)
    _phi, C, _counts = evaluate_constraint_set(blocks, vertices, edges)
    plus = vertices + eta * unflatten_vec3_block(h)
    minus = vertices - eta * unflatten_vec3_block(h)
    phi_p, _, _ = evaluate_constraint_set(blocks, plus, edges)
    phi_m, _, _ = evaluate_constraint_set(blocks, minus, edges)
    fd = (phi_p - phi_m) / (2.0 * eta)
    Ch = C @ h
    rel = safe_norm(fd - Ch) / max(1.0, safe_norm(Ch))
    return rel, rel <= 1.0e-6


def main(argv: List[str]) -> int:
    if len(argv) not in (3, 4):
        print(
            "Usage: python tpe_constraints_oracle.py fixture.json golden.json [mode]",
            file=sys.stderr,
        )
        return 2
    mode = argv[3] if len(argv) == 4 else "length"
    if mode not in ("length", "edgelengths", "point"):
        print(
            f"unknown mode {mode!r}; expected length | edgelengths | point",
            file=sys.stderr,
        )
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
    L0 = total_length(vertices, edges)

    # Mode -> ConstraintSet. Barycenter FIRST in every mode (spec §3.2 row
    # order); targets frozen from the INITIAL geometry (spec §3.5).
    extra_targets: Dict[str, Any] = {}
    if mode == "length":
        blocks = [barycenter_block(x0), total_length_block(L0)]
        row_order = "barycenter rows 0..2, totalLength row 3"
        extra_targets["L0_total_length_target"] = float(L0)
    elif mode == "edgelengths":
        ell0 = edge_lengths(vertices, edges)
        blocks = [barycenter_block(x0), edge_lengths_block(ell0)]
        row_order = "barycenter rows 0..2, edgeLengths rows 3..2+|E| (edge order)"
        extra_targets["ell0_edge_length_targets"] = [float(x) for x in ell0]
    else:  # point
        pin_index = 0
        pin_target = vertices[pin_index].copy()
        blocks = [barycenter_block(x0), point_block(pin_index, pin_target)]
        row_order = "barycenter rows 0..2, point rows 3..5 (pinned vertex x,y,z)"
        extra_targets["pin_vertex_index"] = pin_index
        extra_targets["pin_target"] = [float(x) for x in pin_target]

    phi0, C0, counts = evaluate_constraint_set(blocks, vertices, edges)
    g_tilde, lambdas, residual = solve_constrained_gradient_set(
        vertices, edges, alpha, beta, eps, dE, blocks
    )
    step = line_search_step_set(vertices, edges, alpha, beta, eps, dE, g_tilde, blocks)

    # --- property checks --------------------------------------------------
    fd_rel, fd_ok = fd_jacobian_check(blocks, vertices, edges)
    check("stacked FD Jacobian (central, eta=1e-6)", fd_ok, f"rel = {fd_rel:.3e}")
    check("saddle relative residual <= 1e-10", residual <= 1.0e-10, f"residual = {residual:.3e}")
    slope0 = float(np.dot(flatten_vec3_block(dE), flatten_vec3_block(g_tilde)))
    check("descent positivity dE.g_tilde > 0", slope0 > 0.0, f"dot = {slope0:.3e}")
    cg = C0 @ flatten_vec3_block(g_tilde)
    cg_rel = safe_norm(cg) / max(1.0, safe_norm(flatten_vec3_block(g_tilde)))
    check("constraint compatibility ||C g||/max(1,||g||) <= 1e-10", cg_rel <= 1.0e-10, f"rel = {cg_rel:.3e}")
    check("line search accepted", bool(step["accepted"]), f"tau = {step['tau']}")
    if step["accepted"]:
        check(
            "energy decrease",
            step["energy_after"] < step["energy_before"],
            f"E {step['energy_before']:.9e} -> {step['energy_after']:.9e}",
        )
        newV = np.asarray(step["vertices"], dtype=float)
        if mode == "length":
            L1 = total_length(newV, edges)
            drift = abs(L1 - L0) / max(1.0, L0)
            check("length drift |L-L0|/max(1,L0) <= 1e-8 after step", drift <= 1.0e-8, f"drift = {drift:.3e}")
        elif mode == "edgelengths":
            ell1 = edge_lengths(newV, edges)
            ell0_arr = np.asarray(extra_targets["ell0_edge_length_targets"], dtype=float)
            worst = float(np.max(np.abs(ell1 - ell0_arr) / ell0_arr))
            check(
                "per-edge drift max_I |l_I - l0_I|/l0_I <= 1e-8 after step",
                worst <= 1.0e-8,
                f"max drift = {worst:.3e}",
            )
        else:  # point
            pin = newV[int(extra_targets["pin_vertex_index"])]
            dist = safe_norm(pin - np.asarray(extra_targets["pin_target"], dtype=float))
            check(
                "pin distance ||gamma_pin - target|| <= 1e-8 after step",
                dist <= 1.0e-8,
                f"dist = {dist:.3e}",
            )
        phi1, _, counts1 = evaluate_constraint_set(blocks, newV, edges)
        per_block_ok = constraint_set_converged(blocks, newV, edges, phi1, counts1, 1.0e-10, 1.0e-10)
        check("per-block Phi tolerances hold after step (spec §3.3)", per_block_ok, f"||Phi|| = {safe_norm(phi1):.3e}")

    out: Dict[str, Any] = {
        "conventions": {
            "flattening": "coordinate-block: [x0..xN-1, y0..yN-1, z0..zN-1]",
            "constraint_set": [b["kind"] for b in blocks],
            "row_order": row_order,
            "per_block_projection_tolerance": "spec §3.3 (our invention, not paper)",
            "degenerate_unit_tangent_tol": 1.0e-14,
        },
        "alpha": alpha,
        "beta": beta,
        "epsilon": eps,
        "finite_difference_h": h,
        "energy": float(E),
        "dE": to_jsonable_vec3(dE),
        "dE_flat": [float(x) for x in flatten_vec3_block(dE)],
        "x0_barycenter_target": [float(x) for x in x0],
        "block_kinds": [b["kind"] for b in blocks],
        "block_row_counts": counts,
        "block_scales": [float(b["scale"](vertices, edges)) for b in blocks],
        "Phi_stacked": [float(x) for x in phi0],
        "C_stacked": [[float(x) for x in row] for row in C0],
        "g_tilde": to_jsonable_vec3(g_tilde),
        "g_tilde_flat": [float(x) for x in flatten_vec3_block(g_tilde)],
        "lambda": [float(x) for x in lambdas],
        "saddle_relative_residual": float(residual),
        "gradient_l2_norm": float(l2_curve_norm_vec3(g_tilde, vertices, edges)),
        "line_search_step": step,
        **extra_targets,
    }

    with open(argv[2], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")

    if _FAILURES:
        print(f"\n{len(_FAILURES)} property check(s) FAILED: {_FAILURES}", file=sys.stderr)
        return 1
    print("\nall property checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
