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
    python tpe_constraints_oracle.py fixture.json golden.json [mode] [projection] [penaltyPreset]

mode: "length" (default, M1) = [barycenter, totalLength];
      "edgelengths" (M2)     = [barycenter, edgeLengths];
      "point" (M2)           = [barycenter, point(vertex 0, initial position)];
      "bary" (5C)            = [barycenter] only — the soft-constraint flow
      configuration for penalty goldens (penalties instead of hard length
      constraints, the paper's soft-mode comparison).

penaltyPreset (5C): one of PENALTY_PRESETS ("pen-length" | "pen-diff" |
      "pen-field" | "pen-combo"). Adds the paper's penalty catalog
      (SelfAvoiding.tex lines 762-767) to the OBJECTIVE: analytic penalty
      gradients into dE, penalty energies into the line-search Armijo gate.
      Never touches the constraint rows or the H^s inner product (tex line
      769). Absent => the penalty code is inert and every output is
      byte-identical to before.
      @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md

Tolerance provenance: the projection stopping tolerance 1e-4 is the reference
implementation's backproj_threshold (ythea/repulsive-curves
src/tpe_flow_sc.cpp:15) — reference VALUE fed into OUR per-block scaled rule
(spec §3.3). The original 1e-10 was our invention; gating the frozen mode at
it caused the 2026-07-03 false kill (see oracle/README.md "Projection
tolerance provenance"). The stage-1 oracle (read-only deliverable) keeps its
baked-in 1e-10; its goldens are a frozen 1e-10-era contract.

projection: "reassemble" (default) = reassemble + refactor K at every
      projection iterate (the M1/M2 golden semantics, regenerated 2026-07-03
      at the reference tolerance);
      "frozen" = factor K(gamma_0) ONCE (scipy.linalg.lu_factor) and reuse it
      for the gradient solve and every projection iterate of every tau-trial,
      with Phi evaluated FRESH each iterate (quasi-Newton: frozen metric +
      frozen Jacobian, live residual). Paper-sanctioned: "constraint projection
      with direct solvers comes nearly for free, since a factorization of [the
      constraint saddle] can be reused to solve [the gradient saddle]"
      (SelfAvoiding.tex line 734). Emits `<fixture>-<mode>-frozen.json` and
      additionally gates projection_iterations <= 3 on the accepted step.
      @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 6)
"""
from __future__ import annotations

import json
import math
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

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


# ---------------------------------------------------------------------------
# Penalties (5C): soft objective terms from the paper's catalog,
# SelfAvoiding.tex §Constraints and Potentials lines 762-767. They enter the
# OBJECTIVE only (energy + differential) — never the constraint rows; the H^s
# inner product is unchanged (tex line 769: "lower-order derivatives ... we can
# continue to use the fractional Sobolev inner product without modification").
# Formulas, conventions, and the decision ledger:
# docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2-§3.
# ---------------------------------------------------------------------------

# {"w_length": float, "w_diff": float, "w_field": float, "X": unit np.ndarray}
Penalties = Dict[str, Any]

# Weights are OUR knobs (plan §3, paper gives no values): chosen so each
# penalty visibly perturbs dE on its golden fixture while the line search
# still accepts; the values are recorded in each golden's "penalties" field.
PENALTY_PRESETS: Dict[str, Dict[str, Any]] = {
    # X = [1,0,1] (normalized in normalize_penalties), NOT an axis perpendicular
    # to a planar fixture: with X exactly orthogonal to every tangent the field
    # penalty degenerates to the length penalty (|TxX|^2 = 1, g = T — plan §2.3
    # limit) and the (T.X) terms would be untested. Field goldens use the
    # non-planar helix fixture for the same reason.
    "pen-length": {"w_length": 0.5},
    "pen-diff": {"w_diff": 10.0},
    "pen-field": {"w_field": 0.5, "X": [1.0, 0.0, 1.0]},
    "pen-combo": {"w_length": 0.25, "w_diff": 5.0, "w_field": 0.25, "X": [1.0, 0.0, 1.0]},
}


def normalize_penalties(cfg: Optional[Dict[str, Any]]) -> Optional[Penalties]:
    """Fill weight defaults (0 = off) and normalize X ONCE; ||X|| < 1e-14 =>
    the field penalty is inactive (plan §3 "X handling")."""
    if cfg is None:
        return None
    out: Penalties = {
        "w_length": float(cfg.get("w_length", 0.0)),
        "w_diff": float(cfg.get("w_diff", 0.0)),
        "w_field": float(cfg.get("w_field", 0.0)),
        "X": np.zeros(3, dtype=float),
    }
    X = np.asarray(cfg.get("X", [0.0, 0.0, 0.0]), dtype=float)
    nX = safe_norm(X)
    if nX < 1.0e-14:
        out["w_field"] = 0.0
    else:
        out["X"] = X / nX
    return out


def interior_vertices(n: int, edges: np.ndarray) -> List[Tuple[int, int, int]]:
    """V_int for the length-difference penalty: (v, I_v, J_v) for every vertex
    of degree EXACTLY 2 — paper line 764 "interior" vertices; degree-1
    endpoints and degree>=3 junctions are EXCLUDED. Incident edges ordered by
    ascending edge index (plan §3: value and gradient are swap-symmetric, the
    order is fixed only for determinism)."""
    incident: List[List[int]] = [[] for _ in range(n)]
    for r, (i1, i2) in enumerate(edges):
        incident[int(i1)].append(r)
        incident[int(i2)].append(r)
    return [(v, inc[0], inc[1]) for v, inc in enumerate(incident) if len(inc) == 2]


def _edge_lengths_and_tangents(
    vertices: np.ndarray, edges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Raw lengths (no +eps) and safe-unit tangents: ell < 1e-14 => T = 0, so
    degenerate edges contribute ZERO to every penalty term (plan §2
    conventions — same guard constant as the constraint rows)."""
    m = edges.shape[0]
    ell = np.zeros(m, dtype=float)
    T = np.zeros((m, 3), dtype=float)
    for r, (i1, i2) in enumerate(edges):
        ev = vertices[int(i2)] - vertices[int(i1)]
        ell[r] = safe_norm(ev)
        T[r] = safe_unit(ev)
    return ell, T


def penalty_energy(vertices: np.ndarray, edges: np.ndarray, pen: Penalties) -> float:
    """E_pen = w_len*sum_I ell_I + w_diff*sum_{V_int}(ell_I - ell_J)^2
    + w_field*sum_I ell_I*|T_I x X|^2 (plan §2.1-§2.3; tex lines 763/764/766).
    Explicit cross product so a degenerate T = 0 contributes 0."""
    ell, T = _edge_lengths_and_tangents(vertices, edges)
    total = 0.0
    if pen["w_length"] != 0.0:
        total += pen["w_length"] * float(np.sum(ell))
    if pen["w_diff"] != 0.0:
        acc = 0.0
        for _v, rI, rJ in interior_vertices(vertices.shape[0], edges):
            d = ell[rI] - ell[rJ]
            acc += d * d
        total += pen["w_diff"] * acc
    if pen["w_field"] != 0.0:
        X = pen["X"]
        acc = 0.0
        for r in range(edges.shape[0]):
            c = np.cross(T[r], X)
            acc += ell[r] * float(np.dot(c, c))
        total += pen["w_field"] * acc
    return total


def penalty_gradient(vertices: np.ndarray, edges: np.ndarray, pen: Penalties) -> np.ndarray:
    """Analytic d(E_pen)/dgamma as an (n,3) vec3 array, ascent orientation —
    ADDS to dE before the saddle solve. Per-edge stencils (plan §2, edge
    I=(a,b), e = gamma_b - gamma_a, T = e/ell):
      total length:  -w*T at a, +w*T at b                       (§2.1)
      length diff:   +-2w*(ell_I - ell_J)*T on each edge's ends (§2.2)
      field:         g = (1+(T.X)^2)*T - 2*(T.X)*X, -+w*g       (§2.3;
                     bounded as ell->0, aligned edge => g = 0)
    FD-verified by the embedded property check (rel <= 1e-6)."""
    n = vertices.shape[0]
    g = np.zeros((n, 3), dtype=float)
    ell, T = _edge_lengths_and_tangents(vertices, edges)
    if pen["w_length"] != 0.0:
        w = pen["w_length"]
        for r, (i1, i2) in enumerate(edges):
            g[int(i1)] -= w * T[r]
            g[int(i2)] += w * T[r]
    if pen["w_diff"] != 0.0:
        w = pen["w_diff"]
        for _v, rI, rJ in interior_vertices(n, edges):
            s = 2.0 * w * (ell[rI] - ell[rJ])
            a, b = int(edges[rI][0]), int(edges[rI][1])
            g[a] -= s * T[rI]
            g[b] += s * T[rI]
            a, b = int(edges[rJ][0]), int(edges[rJ][1])
            g[a] += s * T[rJ]
            g[b] -= s * T[rJ]
    if pen["w_field"] != 0.0:
        w = pen["w_field"]
        X = pen["X"]
        for r, (i1, i2) in enumerate(edges):
            if ell[r] < 1.0e-14:
                continue  # degenerate edge: zero contribution (plan §2 conventions)
            u = float(np.dot(T[r], X))
            ge = (1.0 + u * u) * T[r] - 2.0 * u * X
            g[int(i1)] -= w * ge
            g[int(i2)] += w * ge
    return g


def fd_penalty_check(
    vertices: np.ndarray, edges: np.ndarray, pen: Penalties, eta: float = 1.0e-6
) -> Tuple[float, bool]:
    """Central-difference FD check of the analytic penalty gradient along a
    deterministic direction (fd_jacobian_check pattern; plan §5 gate 1e-6)."""
    n3 = 3 * vertices.shape[0]
    h = np.array([math.sin(1.0 + 0.7 * i) for i in range(n3)], dtype=float)
    h /= safe_norm(h)
    hv = unflatten_vec3_block(h)
    ep = penalty_energy(vertices + eta * hv, edges, pen)
    em = penalty_energy(vertices - eta * hv, edges, pen)
    fd = (ep - em) / (2.0 * eta)
    gh = float(np.dot(flatten_vec3_block(penalty_gradient(vertices, edges, pen)), h))
    rel = abs(fd - gh) / max(1.0, abs(gh))
    return rel, rel <= 1.0e-6


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


# Frozen operator: (K, lu_factor(K), m = 3n). K and m are retained so every
# reuse solve can compute the same self-certifying relative residual as
# solve_saddle — never skipped.
Frozen = Tuple[np.ndarray, Any, int]


def build_frozen_saddle(
    vertices: np.ndarray,
    edges: np.ndarray,
    alpha: float,
    beta: float,
    eps: float,
    blocks: List[Block],
) -> Frozen:
    """K(gamma_0) = [[A3(gamma_0), C(gamma_0)^T], [C(gamma_0), 0]] assembled and
    LU-factored ONCE at the step base point gamma_0, then reused for the
    gradient solve AND every projection iterate of every tau-trial.

    Paper-sanctioned (SelfAvoiding.tex line 734): "constraint projection with
    direct solvers comes nearly for free, since a factorization of [the
    constraint saddle] can be reused to solve [the gradient saddle]".
    @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 6)
    """
    _B, _B0, A, _info = assemble_inner_product(vertices, edges, alpha, beta, eps)
    A3 = expand_vector_inner_product(A)
    _phi, C, _counts = evaluate_constraint_set(blocks, vertices, edges)
    k = C.shape[0]
    K = np.block([[A3, C.T], [C, np.zeros((k, k), dtype=float)]])
    fac = la.lu_factor(K, check_finite=True)
    return K, fac, A3.shape[0]


def solve_saddle_frozen(
    frozen: Frozen, rhs_top: np.ndarray, rhs_bottom: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """lu_solve against the frozen factorization. Same relative-residual
    definition as solve_saddle (||K.z - rhs|| / max(1, ||rhs||)), computed on
    EVERY solve against the frozen K — the self-certifying gate survives the
    reuse. @see oracle/tpe_stage1_oracle.py (solve_saddle)"""
    K, fac, m = frozen
    k = K.shape[0] - m
    if rhs_bottom is None:
        rhs_bottom = np.zeros(k, dtype=float)
    rhs = np.concatenate([rhs_top, rhs_bottom])
    sol = la.lu_solve(fac, rhs, check_finite=True)
    residual = safe_norm(K @ sol - rhs) / max(1.0, safe_norm(rhs))
    return sol[:m], sol[m:], residual


def solve_constrained_gradient_set_frozen(
    vertices: np.ndarray,
    edges: np.ndarray,
    alpha: float,
    beta: float,
    eps: float,
    dE_vec3: np.ndarray,
    blocks: List[Block],
) -> Tuple[np.ndarray, np.ndarray, float, Frozen]:
    """Frozen-mode gradient solve: builds the frozen operator at gamma_0 and
    consumes it for [[A3, C^T], [C, 0]][g; lam] = [dE; 0]; returns the operator
    so the line search reuses the SAME factorization for every projection
    iterate (paper line 734, see build_frozen_saddle)."""
    frozen = build_frozen_saddle(vertices, edges, alpha, beta, eps, blocks)
    g_flat, lam, residual = solve_saddle_frozen(frozen, flatten_vec3_block(dE_vec3))
    return unflatten_vec3_block(g_flat), lam, residual, frozen


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
    # 1e-4: reference-impl backproj_threshold (ythea/repulsive-curves
    # src/tpe_flow_sc.cpp:15); see module docstring "Tolerance provenance".
    tol_abs: float = 1.0e-4,
    tol_rel: float = 1.0e-4,
    max_iter: int = 8,
    frozen: Optional[Frozen] = None,
) -> Tuple[np.ndarray, bool, int, float]:
    """Set-generic mirror of the stage-1 project_barycenter: same
    check-before-correct loop (max_iter + 1 checks), same reassembly of A per
    iterate, same failure semantics; only the convergence test is per-block and
    the reported phi_norm is the STACKED ||Phi||_2.

    frozen (Task 6): when given, the per-iterate solve reuses the frozen
    K(gamma_0) factorization (lu_solve, fresh -Phi RHS) instead of
    reassembling — everything else (stopping rule, tolerances, max_iter,
    non-finite -> ok False, exception -> ok False) is IDENTICAL by
    construction: the same loop runs, only the solve line differs.
    @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 6)"""
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
            if frozen is not None:
                # Frozen K(gamma_0), fresh Phi(gamma_q): quasi-Newton correction.
                x_flat, _mu, _res = solve_saddle_frozen(
                    frozen, np.zeros(3 * cur.shape[0], dtype=float), -phi
                )
            else:
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
    frozen: Optional[Frozen] = None,
    penalties: Optional[Penalties] = None,
) -> Dict[str, Any]:
    """Set-generic mirror of the stage-1 line_search_step (identical control
    flow and payloads; projection swapped for project_constraint_set).

    frozen (Task 6): the gradient solve's K(gamma_0) factorization, forwarded
    to project_constraint_set so EVERY projection iterate of EVERY tau-trial
    reuses it (paper line 734 semantics; see build_frozen_saddle).

    penalties (5C): when given, Armijo gates on the TOTAL objective
    E_tpe + E_pen — the energy the flow is minimizing (plan §2.4); the caller
    must have fed the matching dE_total into the gradient solve. None =>
    byte-identical to before (the objective IS energy()).
    @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2.4"""

    def _objective(V: np.ndarray) -> float:
        e = energy(V, edges, alpha, beta, eps)
        if penalties is not None:
            e += penalty_energy(V, edges, penalties)
        return e

    e0 = _objective(vertices)
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
            raw, edges, alpha, beta, eps, blocks, frozen=frozen
        )
        if ok and np.all(np.isfinite(projected)):
            e1 = _objective(projected)
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
    if len(argv) not in (3, 4, 5, 6):
        print(
            "Usage: python tpe_constraints_oracle.py fixture.json golden.json [mode] [projection] [penaltyPreset]",
            file=sys.stderr,
        )
        return 2
    mode = argv[3] if len(argv) >= 4 else "length"
    if mode not in ("length", "edgelengths", "point", "bary"):
        print(
            f"unknown mode {mode!r}; expected length | edgelengths | point | bary",
            file=sys.stderr,
        )
        return 2
    projection = argv[4] if len(argv) >= 5 else "reassemble"
    if projection not in ("reassemble", "frozen"):
        print(
            f"unknown projection {projection!r}; expected reassemble | frozen",
            file=sys.stderr,
        )
        return 2
    preset = argv[5] if len(argv) == 6 else None
    if preset is not None and preset not in PENALTY_PRESETS:
        print(
            f"unknown penalty preset {preset!r}; expected one of {sorted(PENALTY_PRESETS)}",
            file=sys.stderr,
        )
        return 2
    pen = normalize_penalties(PENALTY_PRESETS[preset]) if preset else None

    with open(argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
    vertices = as_vertices(data["vertices"])
    edges = as_edges(data["edges"], vertices.shape[0])
    alpha = float(data.get("alpha", 3.0))
    beta = float(data.get("beta", 6.0))
    eps = float(data.get("epsilon", 1.0e-10))
    h = float(data.get("finite_difference_h", DEFAULT_FD_H))

    E = energy(vertices, edges, alpha, beta, eps)
    dE_tpe = finite_difference_dE(vertices, edges, alpha, beta, eps, h)
    # dE split (plan §3 ledger): FD for the TPE part (stage-1 convention) +
    # ANALYTIC penalty gradient — the TS side mirrors exactly this composition,
    # so goldens agree at 1e-9 without FD-truncation noise in the penalty part.
    dE = dE_tpe + penalty_gradient(vertices, edges, pen) if pen is not None else dE_tpe
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
    elif mode == "point":
        pin_index = 0
        pin_target = vertices[pin_index].copy()
        blocks = [barycenter_block(x0), point_block(pin_index, pin_target)]
        row_order = "barycenter rows 0..2, point rows 3..5 (pinned vertex x,y,z)"
        extra_targets["pin_vertex_index"] = pin_index
        extra_targets["pin_target"] = [float(x) for x in pin_target]
    else:  # bary (5C): barycenter only — the soft-constraint flow for penalties
        blocks = [barycenter_block(x0)]
        row_order = "barycenter rows 0..2 (only block)"

    phi0, C0, counts = evaluate_constraint_set(blocks, vertices, edges)
    if projection == "frozen":
        # Factor K(gamma_0) ONCE; the gradient solve consumes it and the SAME
        # factorization drives every projection iterate of every tau-trial
        # (paper line 734 — see build_frozen_saddle).
        g_tilde, lambdas, residual, frozen = solve_constrained_gradient_set_frozen(
            vertices, edges, alpha, beta, eps, dE, blocks
        )
    else:
        g_tilde, lambdas, residual = solve_constrained_gradient_set(
            vertices, edges, alpha, beta, eps, dE, blocks
        )
        frozen = None
    step = line_search_step_set(
        vertices, edges, alpha, beta, eps, dE, g_tilde, blocks, frozen=frozen, penalties=pen
    )

    # --- property checks --------------------------------------------------
    fd_rel, fd_ok = fd_jacobian_check(blocks, vertices, edges)
    check("stacked FD Jacobian (central, eta=1e-6)", fd_ok, f"rel = {fd_rel:.3e}")
    if pen is not None:
        pen_rel, pen_ok = fd_penalty_check(vertices, edges, pen)
        check("penalty FD gradient (central, eta=1e-6)", pen_ok, f"rel = {pen_rel:.3e}")
        dE_delta = safe_norm(flatten_vec3_block(dE - dE_tpe))
        check(
            "penalty perturbs dE",
            dE_delta > 0.0,
            f"||dE_pen|| = {dE_delta:.3e}",
        )
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
        # Drift gates TRACK the projection stopping rule (reference tolerance
        # 1e-4 into the per-block scaled rule — module docstring "Tolerance
        # provenance"), recomputed independently of the projection internals.
        # Measured values are printed so the README table records ACTUAL drift
        # (typically far below the bound thanks to Newton overshoot); the old
        # fixed 1e-8 gates were only satisfiable at the pre-provenance 1e-10.
        if mode == "length":
            L1 = total_length(newV, edges)
            bound = 1.0e-4 * max(1.0, L1)
            check(
                "length drift |L-L0| <= 1e-4*max(1,L) after step (stopping rule)",
                abs(L1 - L0) <= bound,
                f"|L-L0| = {abs(L1 - L0):.3e} (bound {bound:.3e})",
            )
        elif mode == "edgelengths":
            ell1 = edge_lengths(newV, edges)
            ell0_arr = np.asarray(extra_targets["ell0_edge_length_targets"], dtype=float)
            stacked = float(np.sqrt(np.sum((ell1 - ell0_arr) ** 2)))
            bound = 1.0e-4 * max(1.0, total_length(newV, edges))
            worst = float(np.max(np.abs(ell1 - ell0_arr) / ell0_arr))
            check(
                "edge-length drift ||l - l0||_2 <= 1e-4*max(1,L) after step (stopping rule)",
                stacked <= bound,
                f"||l-l0|| = {stacked:.3e} (bound {bound:.3e}, max per-edge rel {worst:.3e})",
            )
        elif mode == "point":
            pin = newV[int(extra_targets["pin_vertex_index"])]
            target = np.asarray(extra_targets["pin_target"], dtype=float)
            dist = safe_norm(pin - target)
            bound = 1.0e-4 * max(1.0, float(max(safe_norm(v - target) for v in newV)))
            check(
                "pin distance <= 1e-4*max(1,R) after step (stopping rule)",
                dist <= bound,
                f"dist = {dist:.3e} (bound {bound:.3e})",
            )
        phi1, _, counts1 = evaluate_constraint_set(blocks, newV, edges)
        # 1e-4: same reference tolerance as project_constraint_set's defaults.
        per_block_ok = constraint_set_converged(blocks, newV, edges, phi1, counts1, 1.0e-4, 1.0e-4)
        check("per-block Phi tolerances hold after step (spec §3.3)", per_block_ok, f"||Phi|| = {safe_norm(phi1):.3e}")
        if projection == "frozen":
            # The frozen mode's kill gate (plan Task 6): <= 3 is BOTH the
            # paper's stated expectation (SelfAvoiding.tex line 611) and the
            # reference implementation's hard cap (ythea/repulsive-curves
            # src/tpe_flow_sc.cpp:306, `for (int i = 0; i < 3; i++)`). Only
            # meaningful at the reference tolerance — at the pre-provenance
            # 1e-10 this gate false-killed the mode on 2026-07-03.
            iters = int(step["projection_iterations"])
            check(
                "frozen projection iterations <= 3 (reference-impl hard cap)",
                iters <= 3,
                f"iterations = {iters}",
            )

    out: Dict[str, Any] = {
        "conventions": {
            "flattening": "coordinate-block: [x0..xN-1, y0..yN-1, z0..zN-1]",
            "constraint_set": [b["kind"] for b in blocks],
            "row_order": row_order,
            "per_block_projection_tolerance": "rule: spec §3.3 (ours); value 1e-4: reference backproj_threshold (tpe_flow_sc.cpp:15)",
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
    if projection == "frozen":
        # Recorded ONLY in frozen mode so the default reassemble output stays
        # byte-identical to the committed M1/M2 goldens (plan Task 6 gate).
        out["conventions"]["projection"] = "frozen"
    if pen is not None:
        # Recorded ONLY with a penalty preset so preset-less output stays
        # byte-identical to the committed goldens (plan §4 Task 2 gate).
        # "energy" above stays E_tpe (pre-existing meaning); dE/dE_flat above
        # ARE the total differential fed to the solve (plan §2.4).
        pg = penalty_gradient(vertices, edges, pen)
        out["conventions"]["penalties"] = (
            "objective = E_tpe + penalties; dE/dE_flat are the TOTAL differential "
            "(docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2.4)"
        )
        out["penalties"] = {
            "preset": preset,
            "w_length": pen["w_length"],
            "w_diff": pen["w_diff"],
            "w_field": pen["w_field"],
            "X": [float(x) for x in pen["X"]],
        }
        out["penalty_energy_initial"] = float(penalty_energy(vertices, edges, pen))
        out["objective_energy_initial"] = float(E) + out["penalty_energy_initial"]
        out["dE_tpe_flat"] = [float(x) for x in flatten_vec3_block(dE_tpe)]
        out["dE_penalty_flat"] = [float(x) for x in flatten_vec3_block(pg)]

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
