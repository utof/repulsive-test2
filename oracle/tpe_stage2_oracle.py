#!/usr/bin/env python3
"""
Deterministic Stage-2 oracle for the Repulsive Curves fractional Sobolev discretization.

Single-file, numpy/scipy only.  It intentionally keeps both exact dense paths and
low-order BH/BCT/MG paths in one place so TypeScript implementations can diff against
concrete goldens while still testing the approximation layer.

Input JSON:
  {"name": str?, "vertices": [[x,y,z],...], "edges": [[i,j],...],
   "alpha": 3, "beta": 6, "epsilon": 1e-10,
   "theta": 0.5?, "leaf_size": 8?,
   "constraints": {"barycenter": true, "totalLength": false,
                   "edgeLengths": false,
                   "points": [{"vertex": 0, "target": [x,y,z]}] }? }

Output JSON contains dense exact quantities plus BH energy/dE, BCT matvecs, and
geometric-MG solves.  All flattening is coordinate-block:
[x_0..x_{n-1}, y_0..y_{n-1}, z_0..z_{n-1}].
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg

Array = np.ndarray
TANGENT_TOL = 1e-14

# -----------------------------------------------------------------------------
# Basic geometry and flattening
# -----------------------------------------------------------------------------

def asarray_vertices(x) -> Array:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("vertices must have shape (n,3)")
    return a.copy()


def asarray_edges(x) -> Array:
    e = np.asarray(x, dtype=int)
    if e.ndim != 2 or e.shape[1] != 2:
        raise ValueError("edges must have shape (m,2)")
    return e.copy()


def flatten_block(v: Array) -> Array:
    v = np.asarray(v, dtype=float)
    return np.concatenate([v[:, 0], v[:, 1], v[:, 2]])


def unflatten_block(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    n3 = x.size
    if n3 % 3:
        raise ValueError("block vector length not divisible by 3")
    n = n3 // 3
    return np.stack([x[:n], x[n:2*n], x[2*n:]], axis=1)


def block_index(n: int, vertex: int, coord: int) -> int:
    return coord * n + vertex


def edge_geometry(vertices: Array, edges: Array, epsilon: float) -> Dict[str, Array]:
    a = edges[:, 0]
    b = edges[:, 1]
    e = vertices[b] - vertices[a]
    ell_raw = np.linalg.norm(e, axis=1)
    ell_eps = ell_raw + epsilon
    T = np.zeros_like(e)
    mask = ell_raw >= TANGENT_TOL
    T[mask] = e[mask] / ell_raw[mask, None]
    C = 0.5 * (vertices[a] + vertices[b])
    return {"a": a, "b": b, "e": e, "ell_raw": ell_raw, "ell_eps": ell_eps, "T": T, "C": C}


def disjoint_pairs(edges: Array) -> List[List[int]]:
    m = len(edges)
    out: List[List[int]] = []
    sets = [set(map(int, e)) for e in edges]
    for i in range(m):
        out.append([j for j in range(m) if i != j and sets[i].isdisjoint(sets[j])])
    return out


def disjoint_matrix(edges: Array) -> Array:
    m = len(edges)
    D = np.zeros((m, m), dtype=bool)
    sets = [set(map(int, e)) for e in edges]
    for i in range(m):
        for j in range(m):
            D[i, j] = i != j and sets[i].isdisjoint(sets[j])
    return D


def rel_norm(a: Array, b: Array) -> float:
    return float(np.linalg.norm(a - b) / max(1.0, np.linalg.norm(b)))

# -----------------------------------------------------------------------------
# Exact app energy and FD differential
# -----------------------------------------------------------------------------

def calculate_energy(vertices: Array, edges: Array, alpha: float, beta: float, epsilon: float) -> float:
    geom = edge_geometry(vertices, edges, epsilon)
    pairs = disjoint_pairs(edges)
    total = 0.0
    for I, Js in enumerate(pairs):
        i1, i2 = map(int, edges[I])
        eI = geom["e"][I]
        ellI = geom["ell_eps"][I]
        for J in Js:
            j1, j2 = map(int, edges[J])
            ellJ = geom["ell_eps"][J]
            sumK = 0.0
            for i in (i1, i2):
                for j in (j1, j2):
                    d = vertices[i] - vertices[j]
                    d_norm = np.linalg.norm(d) + epsilon
                    c_norm = np.linalg.norm(np.cross(eI, d)) + epsilon
                    sumK += (c_norm ** alpha) / (d_norm ** beta)
            total += 0.25 * (ellI ** (1.0 - alpha)) * ellJ * sumK
    return 0.5 * total


def finite_difference_gradient(func, vertices: Array, h: float = 1e-6) -> Array:
    base = float(func(vertices))
    g = np.zeros_like(vertices)
    for i in range(vertices.shape[0]):
        for c in range(3):
            vp = vertices.copy()
            vp[i, c] += h
            g[i, c] = (float(func(vp)) - base) / h
    return g

# -----------------------------------------------------------------------------
# Exact dense B, B0, A
# -----------------------------------------------------------------------------

def assemble_B_B0(vertices: Array, edges: Array, alpha: float, beta: float, epsilon: float) -> Tuple[Array, Array, Array]:
    if abs(alpha - 3.0) > 1e-14 or abs(beta - 6.0) > 1e-14:
        raise ValueError("this oracle specializes the inner product to alpha=3,beta=6")
    n = len(vertices)
    B = np.zeros((n, n), dtype=float)
    B0 = np.zeros((n, n), dtype=float)
    geom = edge_geometry(vertices, edges, epsilon)
    pairs = disjoint_pairs(edges)
    p = 7.0 / 3.0
    for I, Js in enumerate(pairs):
        ii = [int(edges[I, 0]), int(edges[I, 1])]
        ellI = geom["ell_eps"][I]
        TI = geom["T"][I]
        for J in Js:
            jj = [int(edges[J, 0]), int(edges[J, 1])]
            ellJ = geom["ell_eps"][J]
            TJ = geom["T"][J]
            wij_sum = 0.0
            w0_sum = 0.0
            for vi in ii:
                for vj in jj:
                    d = vertices[vi] - vertices[vj]
                    dn = np.linalg.norm(d) + epsilon
                    wij_sum += 1.0 / (dn ** p)
                    # low-order k_4^2 with target tangent T_I and epsilon after norms
                    crossn = np.linalg.norm(np.cross(TI, d)) + epsilon
                    k24 = (crossn ** 2.0) / (dn ** 4.0)
                    w0_sum += k24 / (dn ** p)
            w = 0.25 * ellI * ellJ * wij_sum
            w0 = 0.25 * ellI * ellJ * w0_sum
            dotTT = float(np.dot(TI, TJ))
            for a in range(2):
                for b in range(2):
                    s = 1.0 if ((a + b) % 2 == 0) else -1.0
                    ia, ib = ii[a], ii[b]
                    ja, jb = jj[a], jj[b]
                    B[ia, ib] += s * w / (ellI * ellI)
                    B[ia, jb] -= s * w * dotTT / (ellI * ellJ)
                    B[ja, jb] += s * w / (ellJ * ellJ)
                    B[ja, ib] -= s * w * dotTT / (ellJ * ellI)
                    B0[ia, ib] += 0.25 * w0
                    B0[ia, jb] -= 0.25 * w0
                    B0[ja, ib] -= 0.25 * w0
                    B0[ja, jb] += 0.25 * w0
    A = B + B0
    return B, B0, A


def expand_A3(A: Array) -> Array:
    n = A.shape[0]
    Z = np.zeros_like(A)
    return np.block([[A, Z, Z], [Z, A, Z], [Z, Z, A]])

# -----------------------------------------------------------------------------
# Constraints
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class ConstraintSpec:
    barycenter: bool = True
    total_length: bool = False
    edge_lengths: bool = False
    points: Tuple[Tuple[int, Tuple[float, float, float]], ...] = ()


def initial_constraint_spec(vertices: Array, edges: Array, raw_spec: Optional[dict]) -> ConstraintSpec:
    if raw_spec is None:
        return ConstraintSpec()
    pts = []
    for p in raw_spec.get("points", []):
        vi = int(p["vertex"])
        target = tuple(map(float, p.get("target", vertices[vi].tolist())))
        pts.append((vi, target))
    return ConstraintSpec(
        barycenter=bool(raw_spec.get("barycenter", True)),
        total_length=bool(raw_spec.get("totalLength", False)),
        edge_lengths=bool(raw_spec.get("edgeLengths", False)),
        points=tuple(pts),
    )


def barycenter_target(vertices: Array, edges: Array) -> Array:
    geom = edge_geometry(vertices, edges, 0.0)
    L = float(np.sum(geom["ell_raw"]))
    if L <= 0.0:
        return np.mean(vertices, axis=0)
    return np.sum(geom["ell_raw"][:, None] * geom["C"], axis=0) / L


def length_targets(vertices: Array, edges: Array) -> Tuple[float, Array]:
    geom = edge_geometry(vertices, edges, 0.0)
    return float(np.sum(geom["ell_raw"])), geom["ell_raw"].copy()


def barycenter_phi_C(vertices: Array, edges: Array, x0: Array) -> Tuple[Array, Array]:
    n = len(vertices)
    geom = edge_geometry(vertices, edges, 0.0)
    phi = np.zeros(3, dtype=float)
    C = np.zeros((3, 3*n), dtype=float)
    for I, (a, b) in enumerate(edges):
        a = int(a); b = int(b)
        ell = geom["ell_raw"][I]
        T = geom["T"][I]
        m = geom["C"][I]
        q = m - x0
        phi += ell * q
        for r in range(3):
            for c in range(3):
                C[r, block_index(n, a, c)] += -T[c] * q[r] + (0.5 * ell if r == c else 0.0)
                C[r, block_index(n, b, c)] += +T[c] * q[r] + (0.5 * ell if r == c else 0.0)
    return phi, C


def total_length_phi_C(vertices: Array, edges: Array, L0: float) -> Tuple[Array, Array]:
    n = len(vertices)
    geom = edge_geometry(vertices, edges, 0.0)
    phi = np.array([L0 - float(np.sum(geom["ell_raw"]))], dtype=float)
    C = np.zeros((1, 3*n), dtype=float)
    for I, (a, b) in enumerate(edges):
        a = int(a); b = int(b)
        T = geom["T"][I]
        # d(L0 - ell) = -T·(db-da) = T·da - T·db
        for c in range(3):
            C[0, block_index(n, a, c)] += T[c]
            C[0, block_index(n, b, c)] -= T[c]
    return phi, C


def edge_lengths_phi_C(vertices: Array, edges: Array, ell0: Array, edge_map: Optional[List[List[int]]] = None) -> Tuple[Array, Array]:
    # edge_map[l] lists original fine edges represented by coarse edge l.  If omitted, one-to-one.
    n = len(vertices)
    geom = edge_geometry(vertices, edges, 0.0)
    if edge_map is None:
        targets = ell0[:len(edges)]
    else:
        targets = np.array([float(np.sum(ell0[idxs])) for idxs in edge_map], dtype=float)
    phi = targets - geom["ell_raw"]
    C = np.zeros((len(edges), 3*n), dtype=float)
    for I, (a, b) in enumerate(edges):
        a = int(a); b = int(b)
        T = geom["T"][I]
        for c in range(3):
            C[I, block_index(n, a, c)] += T[c]
            C[I, block_index(n, b, c)] -= T[c]
    return phi, C


def point_phi_C(vertices: Array, edges: Array, points: Sequence[Tuple[int, Tuple[float, float, float]]], vertex_map: Optional[Dict[int, int]] = None) -> Tuple[Array, Array]:
    n = len(vertices)
    rows = []
    phis = []
    for orig_v, target in points:
        if vertex_map is None:
            if orig_v >= n:
                continue
            v = orig_v
        else:
            if orig_v not in vertex_map:
                # Pinned vertices are forced black by hierarchy construction, so this should not happen.
                continue
            v = vertex_map[orig_v]
        target_arr = np.asarray(target, dtype=float)
        phis.extend((vertices[v] - target_arr).tolist())
        for c in range(3):
            row = np.zeros(3*n, dtype=float)
            row[block_index(n, v, c)] = 1.0
            rows.append(row)
    if not rows:
        return np.zeros(0), np.zeros((0, 3*n))
    return np.asarray(phis, dtype=float), np.vstack(rows)


def eval_constraints(vertices: Array, edges: Array, spec: ConstraintSpec, targets: dict,
                     edge_map: Optional[List[List[int]]] = None,
                     vertex_map: Optional[Dict[int, int]] = None) -> Tuple[Array, Array, List[Tuple[str, int, float]]]:
    phis: List[Array] = []
    Cs: List[Array] = []
    blocks: List[Tuple[str, int, float]] = []
    if spec.barycenter:
        p, C = barycenter_phi_C(vertices, edges, targets["x0"])
        phis.append(p); Cs.append(C)
        geom = edge_geometry(vertices, edges, 0.0)
        L = float(np.sum(geom["ell_raw"]))
        R = float(np.max(np.linalg.norm(vertices - targets["x0"][None, :], axis=1))) if len(vertices) else 0.0
        blocks.append(("barycenter", len(p), L * R))
    if spec.total_length:
        p, C = total_length_phi_C(vertices, edges, targets["L0"])
        phis.append(p); Cs.append(C); blocks.append(("totalLength", 1, max(1.0, targets["L0"])))
    if spec.edge_lengths:
        p, C = edge_lengths_phi_C(vertices, edges, targets["ell0"], edge_map=edge_map)
        phis.append(p); Cs.append(C); blocks.append(("edgeLengths", len(p), max(1.0, float(np.linalg.norm(targets["ell0"])))) )
    if spec.points:
        p, C = point_phi_C(vertices, edges, spec.points, vertex_map=vertex_map)
        phis.append(p); Cs.append(C); blocks.append(("point", len(p), 1.0))
    if not phis:
        return np.zeros(0), np.zeros((0, 3*len(vertices))), []
    return np.concatenate(phis), np.vstack(Cs), blocks

# -----------------------------------------------------------------------------
# Dense saddle solve
# -----------------------------------------------------------------------------

def solve_saddle(A3: Array, C: Array, rhs_top: Array, rhs_bottom: Optional[Array] = None) -> Dict[str, Array | float]:
    m = A3.shape[0]
    k = C.shape[0]
    if rhs_bottom is None:
        rhs_bottom = np.zeros(k, dtype=float)
    K = np.block([[A3, C.T], [C, np.zeros((k, k), dtype=float)]])
    rhs = np.concatenate([rhs_top, rhs_bottom])
    try:
        lu, piv = scipy.linalg.lu_factor(K)
        z = scipy.linalg.lu_solve((lu, piv), rhs)
    except Exception:
        z, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    residual = float(np.linalg.norm(K @ z - rhs) / max(1.0, np.linalg.norm(rhs)))
    return {"x": z[:m], "lambda": z[m:], "residual": residual, "K": K}

# -----------------------------------------------------------------------------
# Line search / projection for exact dense baseline
# -----------------------------------------------------------------------------

def mass_lumped_L2_norm(vertices: Array, edges: Array, x_block: Array) -> float:
    x = unflatten_block(x_block)
    n = len(vertices)
    mass = np.zeros(n, dtype=float)
    geom = edge_geometry(vertices, edges, 0.0)
    for I, (a, b) in enumerate(edges):
        mass[int(a)] += 0.5 * geom["ell_raw"][I]
        mass[int(b)] += 0.5 * geom["ell_raw"][I]
    return float(math.sqrt(max(0.0, np.sum(mass[:, None] * x * x))))


def project_constraints(vertices: Array, edges: Array, alpha: float, beta: float, epsilon: float,
                        spec: ConstraintSpec, targets: dict,
                        tol_abs: float = 1e-10, tol_rel: float = 1e-10, max_iter: int = 8) -> Dict[str, object]:
    cur = vertices.copy()
    for it in range(max_iter + 1):
        phi, C, blocks = eval_constraints(cur, edges, spec, targets)
        ok = True
        offset = 0
        for name, size, scale in blocks:
            block_norm = float(np.linalg.norm(phi[offset:offset+size])); offset += size
            if block_norm > max(tol_abs, tol_rel * scale):
                ok = False; break
        if ok:
            return {"vertices": cur, "ok": True, "iterations": it, "phi_norm": float(np.linalg.norm(phi))}
        if it == max_iter:
            break
        _, _, A = assemble_B_B0(cur, edges, alpha, beta, epsilon)
        A3 = expand_A3(A)
        sol = solve_saddle(A3, C, np.zeros(3*len(cur)), -phi)
        step = unflatten_block(sol["x"])
        if not np.all(np.isfinite(step)):
            return {"vertices": cur, "ok": False, "iterations": it, "phi_norm": float(np.linalg.norm(phi))}
        cur = cur + step
    phi, _, _ = eval_constraints(cur, edges, spec, targets)
    return {"vertices": cur, "ok": False, "iterations": max_iter, "phi_norm": float(np.linalg.norm(phi))}


def dense_step(vertices: Array, edges: Array, alpha: float, beta: float, epsilon: float,
               spec: ConstraintSpec, targets: dict) -> Dict[str, object]:
    E0 = calculate_energy(vertices, edges, alpha, beta, epsilon)
    dE = finite_difference_gradient(lambda V: calculate_energy(V, edges, alpha, beta, epsilon), vertices)
    _, _, A = assemble_B_B0(vertices, edges, alpha, beta, epsilon)
    A3 = expand_A3(A)
    phi, C, _ = eval_constraints(vertices, edges, spec, targets)
    sol = solve_saddle(A3, C, flatten_block(dE), np.zeros(C.shape[0]))
    g = sol["x"]
    gnorm = mass_lumped_L2_norm(vertices, edges, g)
    if gnorm > 0:
        p = g / gnorm
    else:
        p = g.copy()
    c1 = 1e-4; rho = 0.5; tau_min = 1e-12
    slope = float(np.dot(flatten_block(dE), p))
    tau = 1.0
    accepted = False
    best = vertices.copy(); Ebest = E0; proj_info = None
    while tau >= tau_min:
        cand = vertices - tau * unflatten_block(p)
        proj = project_constraints(cand, edges, alpha, beta, epsilon, spec, targets)
        if proj["ok"]:
            Vp = np.asarray(proj["vertices"])
            E1 = calculate_energy(Vp, edges, alpha, beta, epsilon)
            if E1 <= E0 - c1 * tau * slope:
                accepted = True; best = Vp; Ebest = E1; proj_info = proj; break
        tau *= rho
    return {
        "energy0": E0,
        "dE": dE,
        "A": A,
        "C": C,
        "gradient": g,
        "lambda": sol["lambda"],
        "saddle_residual": sol["residual"],
        "gradient_l2h_norm": gnorm,
        "accepted": accepted,
        "tau": tau if accepted else 0.0,
        "vertices": best,
        "energy": Ebest,
        "projection": proj_info if proj_info is not None else {},
        "armijo_slope": slope,
    }

# -----------------------------------------------------------------------------
# BVH for Barnes-Hut energy and BCT
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class BVHNode:
    indices: Array
    left: Optional[int]
    right: Optional[int]
    center: Array
    tangent: Array
    mass: float
    rx: float
    rT: float
    is_leaf: bool


def build_edge_bvh(vertices: Array, edges: Array, epsilon: float, leaf_size: int) -> Tuple[List[BVHNode], Array, Array, Array]:
    geom = edge_geometry(vertices, edges, epsilon)
    C = geom["C"]
    T = geom["T"]
    mass = geom["ell_eps"]
    nodes: List[BVHNode] = []

    def rec(idxs: Array) -> int:
        msum = float(np.sum(mass[idxs]))
        if msum > 0:
            cen = np.sum(mass[idxs, None] * C[idxs], axis=0) / msum
            tan = np.sum(mass[idxs, None] * T[idxs], axis=0) / msum
        else:
            cen = np.mean(C[idxs], axis=0)
            tan = np.mean(T[idxs], axis=0)
        rx = float(np.max(np.linalg.norm(C[idxs] - cen[None, :], axis=1))) if len(idxs) else 0.0
        rT = float(np.max(np.linalg.norm(T[idxs] - tan[None, :], axis=1))) if len(idxs) else 0.0
        node_index = len(nodes)
        nodes.append(BVHNode(idxs.copy(), None, None, cen, tan, msum, rx, rT, True))
        if len(idxs) > leaf_size:
            # split on the coordinate in 6D with largest spread, deterministic median split
            X = np.hstack([C[idxs], T[idxs]])
            spread = X.max(axis=0) - X.min(axis=0)
            dim = int(np.argmax(spread))
            order = np.argsort(X[:, dim], kind="mergesort")
            mid = len(idxs) // 2
            left_idxs = idxs[order[:mid]]
            right_idxs = idxs[order[mid:]]
            if len(left_idxs) and len(right_idxs):
                li = rec(left_idxs); ri = rec(right_idxs)
                nodes[node_index].left = li; nodes[node_index].right = ri; nodes[node_index].is_leaf = False
        return node_index

    rec(np.arange(len(edges), dtype=int))
    return nodes, C, T, mass


def all_disjoint(edge_i: int, node_indices: Array, D: Array) -> bool:
    return bool(np.all(D[edge_i, node_indices]))


def blocks_all_disjoint(target: Array, source: Array, D: Array) -> bool:
    if len(target) == 0 or len(source) == 0:
        return True
    return bool(np.all(D[np.ix_(target, source)]))


def bh_ordered_cluster_term(vertices: Array, geom: Dict[str, Array], I: int, q: Array, mass: float,
                            alpha: float, beta: float, epsilon: float) -> float:
    # Source cluster replaces the two endpoint quadrature of source edges.  Target endpoints remain exact.
    # mass here is sum ell_J^eps over source edges; the 1/4 endpoint quadrature is included below.
    i1, i2 = int(geom["a"][I]), int(geom["b"][I])
    eI = geom["e"][I]
    ellI = geom["ell_eps"][I]
    s = 0.0
    for i in (i1, i2):
        d = vertices[i] - q
        dn = np.linalg.norm(d) + epsilon
        cn = np.linalg.norm(np.cross(eI, d)) + epsilon
        s += (cn ** alpha) / (dn ** beta)
    # exact term has 1/4 over two target and two source endpoints; cluster source represents two endpoints -> 1/2
    return 0.5 * (ellI ** (1.0 - alpha)) * mass * s


def bh_energy(vertices: Array, edges: Array, alpha: float, beta: float, epsilon: float,
              theta: float = 0.5, leaf_size: int = 8) -> Tuple[float, Dict[str, float]]:
    geom = edge_geometry(vertices, edges, epsilon)
    nodes, Cedge, Tedge, mass = build_edge_bvh(vertices, edges, epsilon, leaf_size)
    D = disjoint_matrix(edges)
    visits = 0; approximated = 0; direct_pairs = 0

    def traverse(I: int, ni: int) -> float:
        nonlocal visits, approximated, direct_pairs
        visits += 1
        node = nodes[ni]
        if len(node.indices) == 0:
            return 0.0
        dist = float(np.linalg.norm(Cedge[I] - node.center))
        admiss = (not node.is_leaf and all_disjoint(I, node.indices, D) and dist > 0.0 and
                  (node.rx / dist <= theta) and (node.rT <= theta))
        if admiss:
            approximated += len(node.indices)
            return bh_ordered_cluster_term(vertices, geom, I, node.center, node.mass, alpha, beta, epsilon)
        if node.is_leaf:
            val = 0.0
            for J in node.indices:
                if not D[I, J]:
                    continue
                direct_pairs += 1
                j1, j2 = int(edges[J, 0]), int(edges[J, 1])
                eI = geom["e"][I]
                ellI = geom["ell_eps"][I]
                ellJ = geom["ell_eps"][J]
                i1, i2 = int(edges[I, 0]), int(edges[I, 1])
                sumK = 0.0
                for i in (i1, i2):
                    for j in (j1, j2):
                        d = vertices[i] - vertices[j]
                        dn = np.linalg.norm(d) + epsilon
                        cn = np.linalg.norm(np.cross(eI, d)) + epsilon
                        sumK += (cn ** alpha) / (dn ** beta)
                val += 0.25 * (ellI ** (1.0 - alpha)) * ellJ * sumK
            return val
        return traverse(I, node.left) + traverse(I, node.right)  # type: ignore[arg-type]

    total_ordered = 0.0
    for I in range(len(edges)):
        total_ordered += traverse(I, 0)
    return 0.5 * total_ordered, {"visits": visits, "approximated_edge_refs": approximated, "direct_pairs": direct_pairs}

# -----------------------------------------------------------------------------
# BCT matvecs for high and low edge kernels and A matvec
# -----------------------------------------------------------------------------

def exact_edge_kernel_sym(vertices: Array, edges: Array, epsilon: float, kind: str) -> Array:
    geom = edge_geometry(vertices, edges, epsilon)
    m = len(edges)
    D = disjoint_matrix(edges)
    K = np.zeros((m, m), dtype=float)
    p = 7.0 / 3.0
    for I in range(m):
        ii = [int(edges[I, 0]), int(edges[I, 1])]
        ellI = geom["ell_eps"][I]
        TI = geom["T"][I]
        for J in range(I + 1, m):
            if not D[I, J]:
                continue
            jj = [int(edges[J, 0]), int(edges[J, 1])]
            ellJ = geom["ell_eps"][J]
            TJ = geom["T"][J]
            if kind == "high":
                s = 0.0
                for vi in ii:
                    for vj in jj:
                        dn = np.linalg.norm(vertices[vi] - vertices[vj]) + epsilon
                        s += 1.0 / (dn ** p)
                w = 0.25 * ellI * ellJ * s
                kval = 2.0 * w  # ordered I,J and J,I
            elif kind == "low":
                sIJ = 0.0; sJI = 0.0
                for vi in ii:
                    for vj in jj:
                        d = vertices[vi] - vertices[vj]
                        dn = np.linalg.norm(d) + epsilon
                        sIJ += ((np.linalg.norm(np.cross(TI, d)) + epsilon) ** 2.0) / (dn ** (4.0 + p))
                        sJI += ((np.linalg.norm(np.cross(TJ, -d)) + epsilon) ** 2.0) / (dn ** (4.0 + p))
                kval = 0.25 * ellI * ellJ * (sIJ + sJI)
            else:
                raise ValueError(kind)
            K[I, J] = K[J, I] = kval
    return K


def bct_kernel_node(kind: str, cT: Array, tT: Array, cS: Array, tS: Array, epsilon: float) -> float:
    d = cT - cS
    dn = np.linalg.norm(d) + epsilon
    if kind == "high":
        return 2.0 / (dn ** (7.0 / 3.0))
    if kind == "low":
        c1 = np.linalg.norm(np.cross(tT, d)) + epsilon
        c2 = np.linalg.norm(np.cross(tS, -d)) + epsilon
        return (c1 ** 2.0 + c2 ** 2.0) / (dn ** (4.0 + 7.0 / 3.0))
    raise ValueError(kind)


def bct_matvec_kernel(vertices: Array, edges: Array, epsilon: float, psi: Array, kind: str,
                      theta: float = 0.5, leaf_size: int = 8) -> Tuple[Array, Dict[str, float]]:
    psi = np.asarray(psi, dtype=float)
    m = len(edges)
    vector = psi.ndim == 2
    if psi.shape[0] != m:
        raise ValueError("psi first dimension must equal number of edges")
    nodes, Cedge, Tedge, mass = build_edge_bvh(vertices, edges, epsilon, leaf_size)
    D = disjoint_matrix(edges)
    phi = np.zeros_like(psi, dtype=float)
    leaves = 0; admissible_blocks = 0; direct_blocks = 0; direct_pairs = 0

    def rec(ti: int, si: int):
        nonlocal leaves, admissible_blocks, direct_blocks, direct_pairs
        tn = nodes[ti]; sn = nodes[si]
        dist = float(np.linalg.norm(tn.center - sn.center))
        well = (blocks_all_disjoint(tn.indices, sn.indices, D) and dist > 0.0 and
                (max(tn.rx, sn.rx) / dist <= theta) and (max(tn.rT, sn.rT) <= theta))
        if well:
            leaves += 1; admissible_blocks += 1
            kval = bct_kernel_node(kind, tn.center, tn.tangent, sn.center, sn.tangent, epsilon)
            if vector:
                tmp = np.sum(mass[sn.indices, None] * psi[sn.indices], axis=0)
                phi[tn.indices] += mass[tn.indices, None] * kval * tmp[None, :]
            else:
                tmp = float(np.dot(mass[sn.indices], psi[sn.indices]))
                phi[tn.indices] += mass[tn.indices] * kval * tmp
            return
        if tn.is_leaf and sn.is_leaf:
            leaves += 1; direct_blocks += 1
            for I in tn.indices:
                for J in sn.indices:
                    if not D[I, J]:
                        continue
                    # exact endpoint kernel for direct block
                    kij = exact_edge_kernel_pair(vertices, edges, epsilon, int(I), int(J), kind)
                    if vector:
                        phi[I] += kij * psi[J]
                    else:
                        phi[I] += kij * psi[J]
                    direct_pairs += 1
            return
        # Split the larger/non-leaf side for binary-tree BCT.  Deterministic tie: target first.
        split_t = (not tn.is_leaf) and (sn.is_leaf or len(tn.indices) >= len(sn.indices))
        if split_t:
            rec(tn.left, si)  # type: ignore[arg-type]
            rec(tn.right, si)  # type: ignore[arg-type]
        else:
            rec(ti, sn.left)  # type: ignore[arg-type]
            rec(ti, sn.right)  # type: ignore[arg-type]

    rec(0, 0)
    return phi, {"leaves": leaves, "admissible_blocks": admissible_blocks,
                 "direct_blocks": direct_blocks, "direct_pairs": direct_pairs}


def exact_edge_kernel_pair(vertices: Array, edges: Array, epsilon: float, I: int, J: int, kind: str) -> float:
    if I == J:
        return 0.0
    if set(map(int, edges[I])).intersection(set(map(int, edges[J]))):
        return 0.0
    geom = edge_geometry(vertices, edges, epsilon)
    p = 7.0 / 3.0
    ii = [int(edges[I, 0]), int(edges[I, 1])]
    jj = [int(edges[J, 0]), int(edges[J, 1])]
    ellI = geom["ell_eps"][I]; ellJ = geom["ell_eps"][J]
    TI = geom["T"][I]; TJ = geom["T"][J]
    if kind == "high":
        s = 0.0
        for vi in ii:
            for vj in jj:
                dn = np.linalg.norm(vertices[vi] - vertices[vj]) + epsilon
                s += 1.0 / (dn ** p)
        return 2.0 * 0.25 * ellI * ellJ * s
    if kind == "low":
        sIJ = 0.0; sJI = 0.0
        for vi in ii:
            for vj in jj:
                d = vertices[vi] - vertices[vj]
                dn = np.linalg.norm(d) + epsilon
                sIJ += ((np.linalg.norm(np.cross(TI, d)) + epsilon) ** 2.0) / (dn ** (4.0 + p))
                sJI += ((np.linalg.norm(np.cross(TJ, -d)) + epsilon) ** 2.0) / (dn ** (4.0 + p))
        return 0.25 * ellI * ellJ * (sIJ + sJI)
    raise ValueError(kind)


def edge_average_u(u: Array, edges: Array) -> Array:
    return 0.5 * (u[edges[:, 0]] + u[edges[:, 1]])


def edge_difference_Du(vertices: Array, edges: Array, epsilon: float, u: Array) -> Array:
    geom = edge_geometry(vertices, edges, epsilon)
    du = (u[edges[:, 1]] - u[edges[:, 0]]) / geom["ell_eps"]
    return du[:, None] * geom["T"]


def apply_Dt(vertices: Array, edges: Array, epsilon: float, z: Array) -> Array:
    n = len(vertices)
    geom = edge_geometry(vertices, edges, epsilon)
    out = np.zeros(n, dtype=float)
    for I, (a, b) in enumerate(edges):
        coeff = float(np.dot(z[I], geom["T"][I])) / geom["ell_eps"][I]
        out[int(a)] -= coeff
        out[int(b)] += coeff
    return out


def bct_apply_B_scalar(vertices: Array, edges: Array, epsilon: float, u: Array, kind: str,
                       theta: float, leaf_size: int) -> Tuple[Array, Dict[str, float]]:
    if kind == "low":
        y = edge_average_u(u, edges)
        ones = np.ones(len(edges))
        K1, stats1 = bct_matvec_kernel(vertices, edges, epsilon, ones, "low", theta, leaf_size)
        Ky, stats2 = bct_matvec_kernel(vertices, edges, epsilon, y, "low", theta, leaf_size)
        z = K1 * y - Ky
        out = np.zeros(len(vertices), dtype=float)
        for I, (a, b) in enumerate(edges):
            out[int(a)] += 0.5 * z[I]
            out[int(b)] += 0.5 * z[I]
        stats = {"K1_" + k: v for k, v in stats1.items()}
        stats.update({"Ky_" + k: v for k, v in stats2.items()})
        return out, stats
    if kind == "high":
        y = edge_difference_Du(vertices, edges, epsilon, u)
        ones = np.ones(len(edges))
        K1, stats1 = bct_matvec_kernel(vertices, edges, epsilon, ones, "high", theta, leaf_size)
        Ky, stats2 = bct_matvec_kernel(vertices, edges, epsilon, y, "high", theta, leaf_size)
        z = K1[:, None] * y - Ky
        out = apply_Dt(vertices, edges, epsilon, z)
        stats = {"K1_" + k: v for k, v in stats1.items()}
        stats.update({"Ky_" + k: v for k, v in stats2.items()})
        return out, stats
    raise ValueError(kind)

# -----------------------------------------------------------------------------
# Coarsening/prolongation and geometric multigrid
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class Level:
    vertices: Array
    edges: Array
    orig_vertices: List[int]
    edge_paths: List[List[int]]
    P_to_fine: Optional[Array] = None  # scalar prolongation from this level to previous finer level
    A3: Optional[Array] = None
    C: Optional[Array] = None
    phi: Optional[Array] = None
    blocks: Optional[List[Tuple[str, int, float]]] = None


def adjacency(n: int, edges: Array) -> List[List[Tuple[int, int]]]:
    adj = [[] for _ in range(n)]
    for ei, (a, b) in enumerate(edges):
        a = int(a); b = int(b)
        adj[a].append((b, ei)); adj[b].append((a, ei))
    return adj


def build_coarse_level(level: Level, pinned_orig: set, min_vertices: int = 8) -> Optional[Level]:
    V = level.vertices; E = level.edges; n = len(V)
    if n <= min_vertices:
        return None
    adj = adjacency(n, E)
    deg = np.array([len(x) for x in adj], dtype=int)
    anchors = set(i for i in range(n) if deg[i] != 2)
    for i, orig in enumerate(level.orig_vertices):
        if orig in pinned_orig:
            anchors.add(i)

    black = set(anchors)
    visited_edges = set()
    # Process chains from anchors.  Closed cycles with no anchors are handled later.
    for a in sorted(anchors):
        for nb, ei in adj[a]:
            if ei in visited_edges:
                continue
            path_vertices = [a, nb]
            path_edges = [ei]
            prev, cur = a, nb
            visited_edges.add(ei)
            while cur not in anchors and deg[cur] == 2:
                nxt_candidates = [(x, eidx) for x, eidx in adj[cur] if x != prev]
                if not nxt_candidates:
                    break
                nxt, eidx = nxt_candidates[0]
                if eidx in visited_edges:
                    break
                path_vertices.append(nxt); path_edges.append(eidx)
                visited_edges.add(eidx)
                prev, cur = cur, nxt
            # Alternate along path, endpoints anchors black.  White vertices are odd positions between black vertices.
            for pos, v in enumerate(path_vertices):
                if pos % 2 == 0 or v in anchors:
                    black.add(v)
            if path_vertices[-1] in anchors:
                black.add(path_vertices[-1])
                # Ensure no two white adjacent near odd-length end by forcing penultimate black if needed.
                if len(path_vertices) >= 2 and path_vertices[-2] not in black:
                    black.add(path_vertices[-2])
    # Closed all-degree-2 cycles or unvisited components.
    for start in range(n):
        for nb, ei in adj[start]:
            if ei in visited_edges:
                continue
            cycle = [start]
            prev, cur = start, nb
            visited_edges.add(ei)
            while cur != start:
                cycle.append(cur)
                nxt_candidates = [(x, eidx) for x, eidx in adj[cur] if x != prev]
                if not nxt_candidates:
                    break
                nxt, eidx = nxt_candidates[0]
                if eidx in visited_edges and nxt != start:
                    break
                if eidx not in visited_edges:
                    visited_edges.add(eidx)
                prev, cur = cur, nxt
            # deterministic alternating, force last black for odd cycles
            for pos, v in enumerate(cycle):
                if pos % 2 == 0:
                    black.add(v)
            if len(cycle) % 2 == 1 and len(cycle) > 1:
                black.add(cycle[-1])
    if len(black) == n:
        return None
    black_list = sorted(black)
    coarse_index = {v: i for i, v in enumerate(black_list)}
    P = np.zeros((n, len(black_list)), dtype=float)
    for v in range(n):
        if v in coarse_index:
            P[v, coarse_index[v]] = 1.0
        else:
            # White vertex should have exactly two black neighbors in the reduced path.
            bneigh = [nb for nb, _ in adj[v] if nb in coarse_index]
            if len(bneigh) >= 2:
                P[v, coarse_index[bneigh[0]]] = 0.5
                P[v, coarse_index[bneigh[1]]] = 0.5
            elif len(bneigh) == 1:
                P[v, coarse_index[bneigh[0]]] = 1.0
            else:
                # Fallback: keep as black if something pathological slipped through.
                return None
    # Build coarse edges by walking through fine graph between black vertices.
    coarse_edges = []
    edge_paths: List[List[int]] = []
    seen_pairs = set()
    visited_fine_edges = set()
    for bi in black_list:
        for nb, ei in adj[bi]:
            if ei in visited_fine_edges:
                continue
            path_edges = [ei]
            prev, cur = bi, nb
            visited_fine_edges.add(ei)
            while cur not in coarse_index:
                nxt_candidates = [(x, eidx) for x, eidx in adj[cur] if x != prev]
                if not nxt_candidates:
                    break
                nxt, eidx = nxt_candidates[0]
                if eidx in visited_fine_edges:
                    break
                path_edges.append(eidx); visited_fine_edges.add(eidx)
                prev, cur = cur, nxt
            if cur in coarse_index and cur != bi:
                a = coarse_index[bi]; b = coarse_index[cur]
                key = tuple(sorted((a, b)))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    coarse_edges.append([a, b])
                    # Map to original fine-edge ids by expanding previous level edge paths.
                    orig_path: List[int] = []
                    for pe in path_edges:
                        orig_path.extend(level.edge_paths[pe])
                    edge_paths.append(orig_path)
    if not coarse_edges:
        return None
    coarse_vertices = V[black_list].copy()
    coarse_orig = [level.orig_vertices[i] for i in black_list]
    return Level(coarse_vertices, np.asarray(coarse_edges, dtype=int), coarse_orig, edge_paths, P_to_fine=P)


def block_prolong(P: Array) -> Array:
    Z = np.zeros_like(P)
    return np.block([[P, Z, Z], [Z, P, Z], [Z, Z, P]])


def build_hierarchy(vertices: Array, edges: Array, spec: ConstraintSpec, max_levels: int = 16, min_vertices: int = 8) -> List[Level]:
    pinned = {p[0] for p in spec.points}
    level0 = Level(vertices.copy(), edges.copy(), list(range(len(vertices))), [[i] for i in range(len(edges))])
    levels = [level0]
    while len(levels) < max_levels:
        nxt = build_coarse_level(levels[-1], pinned, min_vertices=min_vertices)
        if nxt is None:
            break
        levels.append(nxt)
    return levels


def projector(C: Array) -> Tuple[Array, Array]:
    m = C.shape[1]
    if C.shape[0] == 0:
        return np.eye(m), np.zeros((m, 0))
    G = C @ C.T
    # SVD/pinv for rank robustness.
    Gpinv = np.linalg.pinv(G, rcond=1e-12)
    Cdag = C.T @ Gpinv
    Pi = np.eye(m) - Cdag @ C
    return Pi, Cdag


def prepare_levels(levels: List[Level], alpha: float, beta: float, epsilon: float,
                   spec: ConstraintSpec, targets: dict) -> None:
    for L in levels:
        _, _, A = assemble_B_B0(L.vertices, L.edges, alpha, beta, epsilon)
        L.A3 = expand_A3(A)
        # Coarse edge-length constraints use aggregate target path lengths.
        vertex_map = {orig: i for i, orig in enumerate(L.orig_vertices)}
        phi, C, blocks = eval_constraints(L.vertices, L.edges, spec, targets, edge_map=L.edge_paths, vertex_map=vertex_map)
        L.phi = phi; L.C = C; L.blocks = blocks


def direct_projected_solve(A: Array, C: Array, rhs: Array) -> Array:
    sol = solve_saddle(A, C, rhs, np.zeros(C.shape[0]))
    return np.asarray(sol["x"], dtype=float)


def projected_residual(A: Array, C: Array, x: Array, rhs: Array) -> Tuple[Array, float]:
    Pi, _ = projector(C)
    r = Pi @ (rhs - A @ x)
    denom = max(1.0, np.linalg.norm(Pi @ rhs))
    return r, float(np.linalg.norm(r) / denom)


def smooth_projected_cg(A: Array, C: Array, x: Array, rhs: Array, steps: int) -> Array:
    Pi, _ = projector(C)
    # CG on H = Pi A Pi in the constraint tangent subspace, fixed small number of steps.
    x = Pi @ x
    r = Pi @ (rhs - A @ x)
    p = r.copy()
    rs = float(np.dot(r, r))
    if rs == 0.0:
        return x
    for _ in range(steps):
        Ap = Pi @ (A @ (Pi @ p))
        denom = float(np.dot(p, Ap))
        if abs(denom) < 1e-300:
            break
        a = rs / denom
        x = Pi @ (x + a * p)
        r = r - a * Ap
        rs_new = float(np.dot(r, r))
        if rs_new <= 1e-30:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return Pi @ x


def v_cycle(levels: List[Level], li: int, x: Array, rhs: Array, nu_pre: int, nu_post: int) -> Array:
    L = levels[li]
    A = L.A3; C = L.C
    assert A is not None and C is not None
    if li == len(levels) - 1 or len(L.vertices) <= 8:
        return direct_projected_solve(A, C, rhs)
    x = smooth_projected_cg(A, C, x, rhs, nu_pre)
    r, _ = projected_residual(A, C, x, rhs)
    P = levels[li+1].P_to_fine
    assert P is not None
    P3 = block_prolong(P)
    rhs_c = P3.T @ r
    Cc = levels[li+1].C
    Pic, _ = projector(Cc)
    rhs_c = Pic @ rhs_c
    ec0 = np.zeros_like(rhs_c)
    ec = v_cycle(levels, li + 1, ec0, rhs_c, nu_pre, nu_post)
    Pif, _ = projector(C)
    x = Pif @ (x + P3 @ ec)
    x = smooth_projected_cg(A, C, x, rhs, nu_post)
    return x


def mg_projected_solve(levels: List[Level], rhs: Array, tol: float = 1e-10, max_cycles: int = 50,
                       nu_pre: int = 2, nu_post: int = 2) -> Dict[str, object]:
    L0 = levels[0]
    A0 = L0.A3; C0 = L0.C
    assert A0 is not None and C0 is not None
    x = np.zeros_like(rhs)
    residuals = []
    for it in range(max_cycles + 1):
        _, rn = projected_residual(A0, C0, x, rhs)
        residuals.append(rn)
        if rn <= tol:
            break
        x = v_cycle(levels, 0, x, rhs, nu_pre, nu_post)
    # If V-cycles stagnate before 1e-10 on a tiny/pathological fixture, perform one direct correction.
    # This preserves a self-certifying final residual while still emitting the MG cycle history.
    _, rn = projected_residual(A0, C0, x, rhs)
    used_direct_cleanup = False
    if rn > tol:
        dx = direct_projected_solve(A0, C0, rhs - A0 @ x)
        Pi, _ = projector(C0)
        x = Pi @ (x + dx)
        used_direct_cleanup = True
        _, rn = projected_residual(A0, C0, x, rhs)
        residuals.append(rn)
    return {"x": x, "iterations": len(residuals) - 1, "residual": rn,
            "residual_history": residuals, "used_direct_cleanup": used_direct_cleanup}


def mg_saddles(vertices: Array, edges: Array, alpha: float, beta: float, epsilon: float,
               spec: ConstraintSpec, targets: dict, dE_block: Array) -> Dict[str, object]:
    levels = build_hierarchy(vertices, edges, spec)
    prepare_levels(levels, alpha, beta, epsilon, spec, targets)
    A0 = levels[0].A3; C0 = levels[0].C; phi0 = levels[0].phi
    assert A0 is not None and C0 is not None and phi0 is not None
    grad_mg = mg_projected_solve(levels, dE_block)
    dense_grad = solve_saddle(A0, C0, dE_block, np.zeros(C0.shape[0]))
    # projection solve: z=Cdag(-phi), solve y in null with rhs=A z, x=z-y
    Pi, Cdag = projector(C0)
    z = Cdag @ (-phi0) if C0.shape[0] else np.zeros(A0.shape[0])
    rhs_proj = A0 @ z
    y_mg = mg_projected_solve(levels, rhs_proj)
    x_proj = z - y_mg["x"]
    dense_proj = solve_saddle(A0, C0, np.zeros(A0.shape[0]), -phi0)
    K = np.block([[A0, C0.T], [C0, np.zeros((C0.shape[0], C0.shape[0]))]])
    rhs_g = np.concatenate([dE_block, np.zeros(C0.shape[0])])
    lam_g = np.linalg.lstsq(C0 @ C0.T, C0 @ (dE_block - A0 @ grad_mg["x"]), rcond=None)[0] if C0.shape[0] else np.zeros(0)
    zg = np.concatenate([grad_mg["x"], lam_g])
    saddle_res_g = float(np.linalg.norm(K @ zg - rhs_g) / max(1.0, np.linalg.norm(rhs_g))) if C0.shape[0] else float(np.linalg.norm(A0 @ grad_mg["x"] - dE_block) / max(1.0, np.linalg.norm(dE_block)))
    rhs_p = np.concatenate([np.zeros(A0.shape[0]), -phi0])
    lam_p = np.linalg.lstsq(C0 @ C0.T, C0 @ (-A0 @ x_proj), rcond=None)[0] if C0.shape[0] else np.zeros(0)
    zp = np.concatenate([x_proj, lam_p])
    saddle_res_p = float(np.linalg.norm(K @ zp - rhs_p) / max(1.0, np.linalg.norm(rhs_p))) if C0.shape[0] else float(np.linalg.norm(A0 @ x_proj) / max(1.0, np.linalg.norm(rhs_p)))
    return {
        "levels": [{"n_vertices": len(L.vertices), "n_edges": len(L.edges), "n_constraints": int(L.C.shape[0])} for L in levels],
        "gradient": {"x": grad_mg["x"], "projected_residual": grad_mg["residual"],
                     "saddle_residual": saddle_res_g, "iterations": grad_mg["iterations"],
                     "residual_history": grad_mg["residual_history"],
                     "used_direct_cleanup": grad_mg["used_direct_cleanup"],
                     "dense_rel_diff": rel_norm(grad_mg["x"], dense_grad["x"])},
        "projection": {"x": x_proj, "projected_residual": y_mg["residual"],
                       "saddle_residual": saddle_res_p, "iterations": y_mg["iterations"],
                       "residual_history": y_mg["residual_history"],
                       "used_direct_cleanup": y_mg["used_direct_cleanup"],
                       "dense_rel_diff": rel_norm(x_proj, dense_proj["x"]),
                       "phi_norm": float(np.linalg.norm(phi0))},
    }

# -----------------------------------------------------------------------------
# Optional catalog penalty differentials
# -----------------------------------------------------------------------------

def total_length_grad(vertices: Array, edges: Array) -> Array:
    g = np.zeros_like(vertices)
    geom = edge_geometry(vertices, edges, 0.0)
    for I, (a, b) in enumerate(edges):
        a = int(a); b = int(b); T = geom["T"][I]
        g[a] -= T; g[b] += T
    return g


def length_difference_grad(vertices: Array, edges: Array) -> Array:
    # Sum over degree-2 vertices only; choose the two incident edges in ascending edge index order.
    g = np.zeros_like(vertices)
    geom = edge_geometry(vertices, edges, 0.0)
    adj = adjacency(len(vertices), edges)
    for v, inc in enumerate(adj):
        if len(inc) != 2:
            continue
        inc_sorted = sorted(inc, key=lambda x: x[1])
        e0 = inc_sorted[0][1]; e1 = inc_sorted[1][1]
        delta = geom["ell_raw"][e0] - geom["ell_raw"][e1]
        # gradient of ell for an edge
        for sign, ei in [(+1.0, e0), (-1.0, e1)]:
            a, b = map(int, edges[ei]); T = geom["T"][ei]
            coeff = 2.0 * delta * sign
            g[a] -= coeff * T
            g[b] += coeff * T
    return g


def field_potential_energy_and_grad_fd(vertices: Array, edges: Array, field=(1.0, 0.0, 0.0), h=1e-6) -> Tuple[float, Array]:
    # Deterministic FD reference for constant unit field X.  Analytical formula is in the written answer.
    X = np.asarray(field, dtype=float)
    X = X / max(1e-300, np.linalg.norm(X))
    def E(V):
        geom = edge_geometry(V, edges, 0.0)
        val = 0.0
        for I in range(len(edges)):
            val += geom["ell_raw"][I] * float(np.dot(np.cross(geom["T"][I], X), np.cross(geom["T"][I], X)))
        return val
    return float(E(vertices)), finite_difference_gradient(E, vertices, h=h)

# -----------------------------------------------------------------------------
# JSON emission
# -----------------------------------------------------------------------------

def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items() if k != "K"}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def deterministic_probe(n: int) -> Array:
    idx = np.arange(n, dtype=float)
    return np.sin(0.37 * (idx + 1.0)) + 0.25 * np.cos(1.17 * (idx + 1.0))


def run_oracle(data: dict) -> dict:
    vertices = asarray_vertices(data["vertices"])
    edges = asarray_edges(data["edges"])
    alpha = float(data.get("alpha", 3.0)); beta = float(data.get("beta", 6.0)); epsilon = float(data.get("epsilon", 1e-10))
    theta = float(data.get("theta", 0.5)); leaf_size = int(data.get("leaf_size", 4))
    spec = initial_constraint_spec(vertices, edges, data.get("constraints"))
    targets = {"x0": barycenter_target(vertices, edges)}
    L0, ell0 = length_targets(vertices, edges)
    targets["L0"] = L0; targets["ell0"] = ell0

    t0 = time.perf_counter()
    E = calculate_energy(vertices, edges, alpha, beta, epsilon)
    dE = finite_difference_gradient(lambda V: calculate_energy(V, edges, alpha, beta, epsilon), vertices)
    B, B0, A = assemble_B_B0(vertices, edges, alpha, beta, epsilon)
    A3 = expand_A3(A)
    phi, C, blocks = eval_constraints(vertices, edges, spec, targets)
    dense = solve_saddle(A3, C, flatten_block(dE), np.zeros(C.shape[0]))
    step = dense_step(vertices, edges, alpha, beta, epsilon, spec, targets)

    # BH at theta and at half theta for monotone error checks.
    bh_results = {}
    for th in [theta, theta / 2.0]:
        Eb, stats = bh_energy(vertices, edges, alpha, beta, epsilon, th, leaf_size)
        gb = finite_difference_gradient(lambda V, th=th: bh_energy(V, edges, alpha, beta, epsilon, th, leaf_size)[0], vertices)
        bh_results[f"theta_{th:g}"] = {"energy": Eb, "dE": gb, "stats": stats,
                                       "energy_rel_error": abs(Eb - E) / max(1.0, abs(E)),
                                       "dE_rel_error_vs_exact_fd": rel_norm(flatten_block(gb), flatten_block(dE))}

    # BCT edge-kernel matvecs and induced B/B0 scalar matvecs.
    m = len(edges); n = len(vertices)
    psi_edge = deterministic_probe(m)
    psi_edge_vec = np.stack([psi_edge, 0.3 * psi_edge[::-1], np.cos(np.arange(m) + 0.4)], axis=1)
    bct = {}
    for kind in ["high", "low"]:
        Kexact = exact_edge_kernel_sym(vertices, edges, epsilon, kind)
        Kpsi_bct, st = bct_matvec_kernel(vertices, edges, epsilon, psi_edge, kind, theta, leaf_size)
        Kpsi_exact = Kexact @ psi_edge
        Kvec_bct, stv = bct_matvec_kernel(vertices, edges, epsilon, psi_edge_vec, kind, theta, leaf_size)
        Kvec_exact = Kexact @ psi_edge_vec
        bct[kind] = {"Kpsi": Kpsi_bct, "Kpsi_exact": Kpsi_exact,
                     "rel_error": rel_norm(Kpsi_bct, Kpsi_exact),
                     "Kvec_rel_error": rel_norm(Kvec_bct.reshape(-1), Kvec_exact.reshape(-1)),
                     "stats": st, "vector_stats": stv}
    u = deterministic_probe(n)
    Bu_bct, stH = bct_apply_B_scalar(vertices, edges, epsilon, u, "high", theta, leaf_size)
    B0u_bct, stL = bct_apply_B_scalar(vertices, edges, epsilon, u, "low", theta, leaf_size)
    bct["B_matvec"] = {"u": u, "Bu_bct": Bu_bct, "Bu_exact": B @ u,
                       "rel_error": rel_norm(Bu_bct, B @ u), "stats": stH}
    bct["B0_matvec"] = {"u": u, "B0u_bct": B0u_bct, "B0u_exact": B0 @ u,
                        "rel_error": rel_norm(B0u_bct, B0 @ u), "stats": stL}

    mg = mg_saddles(vertices, edges, alpha, beta, epsilon, spec, targets, flatten_block(dE))
    elapsed = time.perf_counter() - t0

    props = {
        "sym_B_rel": rel_norm(B, B.T),
        "sym_B0_rel": rel_norm(B0, B0.T),
        "A_min_eig": float(np.linalg.eigvalsh(A).min()) if A.size else 0.0,
        "A_norm2": float(np.linalg.norm(A, 2)) if A.size else 0.0,
        "A_constant_null_rel": float(np.linalg.norm(A @ np.ones(n)) / max(1.0, np.linalg.norm(A, 2))) if n else 0.0,
        "dense_saddle_residual": dense["residual"],
        "dense_constraint_velocity_norm": float(np.linalg.norm(C @ dense["x"])) if C.shape[0] else 0.0,
        "descent_dot": float(np.dot(dense["x"], flatten_block(dE))),
        "mg_gradient_dense_rel_diff": mg["gradient"]["dense_rel_diff"],
        "mg_projection_dense_rel_diff": mg["projection"]["dense_rel_diff"],
    }

    return to_jsonable({
        "name": data.get("name", "unnamed"),
        "conventions": {
            "alpha": alpha, "beta": beta, "epsilon": epsilon,
            "s": (beta - 1.0) / alpha, "sigma": (beta - 1.0) / alpha - 1.0,
            "two_sigma_plus_1": 2.0 * ((beta - 1.0) / alpha - 1.0) + 1.0,
            "flattening": "coordinate-block",
            "energy_pair_factor": 0.5,
            "inner_product_pair_factor": 1.0,
            "theta": theta, "leaf_size": leaf_size,
        },
        "energy": E,
        "dE": dE,
        "B": B,
        "B0": B0,
        "A": A,
        "A3": A3,
        "constraints": {"phi": phi, "C": C, "blocks": blocks, "targets": targets},
        "dense_saddle": {"x": dense["x"], "lambda": dense["lambda"], "residual": dense["residual"]},
        "accepted_step": step,
        "barnes_hut": bh_results,
        "bct": bct,
        "mg": mg,
        "penalties": {
            "total_length_grad": total_length_grad(vertices, edges),
            "length_difference_grad": length_difference_grad(vertices, edges),
            "field_constant_x": dict(zip(["energy", "grad"], field_potential_energy_and_grad_fd(vertices, edges))),
        },
        "properties": props,
        "elapsed_seconds": elapsed,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json")
    ap.add_argument("output_json")
    args = ap.parse_args()
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = run_oracle(data)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
