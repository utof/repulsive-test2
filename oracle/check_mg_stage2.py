#!/usr/bin/env python3
"""Stage-2 MG convergence + divergence-guard checks (issue utof/repulsive-test2#5).

Why: the 2026-07-04 three-auditor audit found pure MG V(2,2) diverges on open
chains/junctions under REDISCRETIZED coarse operators (spectrally inconsistent
with the fine Galerkin form at boundaries; two-grid rho=8.4 with an EXACT coarse
solve), and the dense cleanup then "self-certified" residuals ~1e67 with the
success flag set. This script is the regression gate for the pre-registered fix:

  1. Galerkin coarse operators (A_c = P3^T A_f P3) converge on every topology
     class we ship: open chain (subdivided helix n=59 + synthetic n=61),
     junction network (junction-y subdivided twice, n=50), closed loop (knot).
  2. mg_projected_solve raises MGDivergenceError instead of silently returning
     a post-divergence "certified" answer (exercised via coarse_op="rediscretize",
     retained for exactly this A/B purpose).
  3. Spectral-pencil sanity: max generalized eigenvalue of (P3^T A_f P3, A_c)
     on ker(C_c) is identically 1 under Galerkin coarse operators.

Run: uv run --with numpy --with scipy python oracle/check_mg_stage2.py
Exit 0 = all checks pass; non-zero with a FAIL line otherwise.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tpe_stage2_oracle as O

FIXDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def load_fixture(name):
    with open(os.path.join(FIXDIR, f"{name}.json"), "r", encoding="utf-8") as f:
        d = json.load(f)
    return O.asarray_vertices(d["vertices"]), O.asarray_edges(d["edges"])


def subdivide(V, E):
    pts = [v for v in V]
    newE = []
    for a, b in E:
        m = 0.5 * (V[int(a)] + V[int(b)])
        mi = len(pts)
        pts.append(m)
        newE.append([int(a), mi])
        newE.append([mi, int(b)])
    return np.asarray(pts), np.asarray(newE, dtype=int)


def open_chain(n):
    t = np.linspace(0.0, 1.0, n)
    V = np.stack([t, 0.15 * np.sin(2 * np.pi * t), 0.1 * np.cos(3 * np.pi * t)], axis=1)
    E = np.asarray([[i, i + 1] for i in range(n - 1)], dtype=int)
    return V, E


def prepare(V, E, coarse_op, alpha=3.0, beta=6.0, eps=1e-10):
    spec = O.ConstraintSpec()  # barycenter only — the committed fixtures' default
    targets = {"x0": O.barycenter_target(V, E)}
    L0, ell0 = O.length_targets(V, E)
    targets["L0"] = L0
    targets["ell0"] = ell0
    levels = O.build_hierarchy(V, E, spec)
    O.prepare_levels(levels, alpha, beta, eps, spec, targets, coarse_op=coarse_op)
    return levels


def probe_rhs(levels):
    A0, C0 = levels[0].A3, levels[0].C
    n3 = A0.shape[0]
    v = np.sin(np.arange(1, n3 + 1) * 0.7) + 0.3 * np.cos(np.arange(n3) * 1.3)
    Pi, _ = O.projector(C0)
    return Pi @ v


def pencil_max_eig(levels, li):
    import scipy.linalg as sla

    P3 = O.block_prolong(levels[li].P_to_fine)
    G = P3.T @ levels[li - 1].A3 @ P3
    Ac = levels[li].A3
    Cc = levels[li].C
    if Cc.shape[0]:
        _, s, Vt = np.linalg.svd(Cc)
        rank = int(np.sum(s > 1e-10 * s[0])) if s.size else 0
        Z = Vt[rank:].T
    else:
        Z = np.eye(Ac.shape[0])
    Gz = 0.5 * ((Z.T @ G @ Z) + (Z.T @ G @ Z).T)
    Az = 0.5 * ((Z.T @ Ac @ Z) + (Z.T @ Ac @ Z).T)
    reg = 1e-14 * max(1.0, np.linalg.norm(Az, 2)) * np.eye(len(Az))
    return float(np.max(sla.eigh(Gz, Az + reg, eigvals_only=True)))


failures = []


def check(label, ok, detail=""):
    tag = "ok  " if ok else "FAIL"
    print(f"{tag} {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def main():
    helV, helE = load_fixture("helix")
    jV, jE = load_fixture("junction-y")
    cases = {
        "helix-sub(n=59,open)": subdivide(helV, helE),
        "chain(n=61,open)": open_chain(61),
        "junction-y-sub2(n=50)": subdivide(*subdivide(jV, jE)),
        "knot(closed)": load_fixture("knot"),
    }
    TOL = 1e-10

    # 1. Galerkin coarse operators converge on every topology class.
    for name, (V, E) in cases.items():
        levels = prepare(V, E, coarse_op="galerkin")
        rhs = probe_rhs(levels)
        try:
            res = O.mg_projected_solve(levels, rhs, tol=TOL)
        except Exception as exc:  # divergence guard must NOT fire here
            check(f"galerkin converges [{name}]", False, f"raised {type(exc).__name__}: {exc}")
            continue
        rn = float(res["residual"])
        xn = float(np.linalg.norm(res["x"]))
        check(
            f"galerkin converges [{name}]",
            np.isfinite(rn) and rn <= 100 * TOL and xn < 1e3,
            f"residual={rn:.3e} |x|={xn:.3e} iters={res['iterations']} cleanup={res['used_direct_cleanup']}",
        )

    # 2. Divergence guard: rediscretized coarse ops on an open chain must fail
    #    loudly, never return a silently "certified" post-divergence answer.
    for name in ["chain(n=61,open)", "helix-sub(n=59,open)"]:
        V, E = cases[name]
        levels = prepare(V, E, coarse_op="rediscretize")
        rhs = probe_rhs(levels)
        try:
            res = O.mg_projected_solve(levels, rhs, tol=TOL)
            rn = float(res["residual"])
            # Reaching here with a bad residual is exactly the issue-#5 bug.
            check(f"divergence guard raises [{name}]", rn <= 100 * TOL,
                  f"returned residual={rn:.3e} with no exception")
        except O.MGDivergenceError:
            check(f"divergence guard raises [{name}]", True)

    # 2b. Nested-iteration init (paper line 1109): default init converges on
    #     every topology class and never needs materially more cycles than the
    #     zero init it replaced (SelfAvoiding.tex: "works much better than
    #     starting with the zero vector"). @issue utof/repulsive-test2#5 item 3
    for name, (V, E) in cases.items():
        levels = prepare(V, E, coarse_op="galerkin")
        rhs = probe_rhs(levels)
        try:
            res_n = O.mg_projected_solve(levels, rhs, tol=TOL, init="nested")
            res_z = O.mg_projected_solve(levels, rhs, tol=TOL, init="zero")
        except Exception as exc:
            check(f"nested init converges [{name}]", False, f"raised {type(exc).__name__}: {exc}")
            continue
        rn = float(res_n["residual"])
        ok = np.isfinite(rn) and rn <= 100 * TOL and res_n["iterations"] <= res_z["iterations"] + 2
        check(
            f"nested init converges, no slower than zero init [{name}]",
            ok,
            f"iters nested={res_n['iterations']} zero={res_z['iterations']} residual={rn:.3e}",
        )

    # 3. Pencil sanity: Galerkin makes the level-1 pencil identically 1.
    for name in ["chain(n=61,open)", "junction-y-sub2(n=50)"]:
        V, E = cases[name]
        levels = prepare(V, E, coarse_op="galerkin")
        ev = pencil_max_eig(levels, 1)
        check(f"galerkin pencil == 1 [{name}]", abs(ev - 1.0) < 1e-6, f"max-eig={ev:.6f}")
        levels_r = prepare(V, E, coarse_op="rediscretize")
        ev_r = pencil_max_eig(levels_r, 1)
        check(f"rediscretized pencil > 1 documented [{name}]", ev_r > 1.5, f"max-eig={ev_r:.3f}")

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED")
        return 1
    print("all stage-2 MG checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
