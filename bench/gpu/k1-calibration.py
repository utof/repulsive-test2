#!/usr/bin/env python3
"""
K1 gate calibration for the WebGPU solver Phase 2a milestone.

Produces, per fixture N, the four measured quantities the Phase 2 spec's K1
gate and precision model are built from — so that the pre-registered bars have
provenance inside the repository rather than in a scratch directory:

  alpha  the backward error a CORRECT f32 Cholesky of A_sigma achieves,
         ||L L^T - A_sigma||_F / ||A_sigma||_F, LAPACK `spotrf` via
         numpy.linalg.cholesky on the f32-cast matrix.
  beta   the LARGEST backward error the spec's IR loop still survives T4 with
         (rel residual <= 1e-10 within <= 4 refinements), found by bisection on
         a perturbation `Lref + s*E` of the exact f64 factor.  beta is
         DIRECTION-DEPENDENT: it is measured over five fixed perturbation
         directions (seeds 0..4) and reported as min / median / max.  The spec
         derives its bars from the MINIMUM, never a fit or a median, because a
         single direction understates how bad a real kernel may be.
  kappa  kappa_2(A_sigma) and kappa_2(K), so the "A_sigma is within 2% of K"
         proxy the precision model leans on is a measurement, not an assumption
         (and so kappa_2(K) cross-checks the committed [G6] table).
  r0/rho the initial relative residual and the IR contraction factor for both
         right-hand-side shapes (gradient [dE;0], projection [0;-phi]) driven
         by a genuine f32 factor — the "one precision model" of spec 4.2.

  cg     unpreconditioned and Jacobi-preconditioned CG iterations to CG_TOL on
         A_sigma — variant (b)'s cost driver in spec 3, and therefore an input
         to the `G5-confirm` gate's own thresholds.  Also RHS-dependent, so it
         is measured over the same five directions and reported min/med/max;
         the spec prices (b) at its MINIMUM, because a faster (b) is the
         conservative direction for a gate that decides against (a).

Also reported: the boot self-test's forward metric max|L32-L64|/max|L64|, whose
1e-5 bar gates GPU-solve availability at boot.

Why the bars are NOT read off a fit: `beta * kappa_2(K)` is not a law.  Against
the minimum beta it measures 0.034 / 0.107 / 0.154 / 0.205 at N=64/240/480/960
— a 6.1x spread — so extrapolating beta from a "0.177/kappa" fit is 5.3x
anti-conservative exactly at N=64, where the fit was the only source.  Every N
in the spec's K1 table is therefore bisected here directly.

Uses (imports, does not copy) the same oracle machinery as
`oracle/check_kappa_peredge.py`, and the same trefoil parametrization as
`src/core/fixtures.ts` `trefoil(n)`, so these numbers describe the milestone's
real fixture.

Run (repo root; no numpy/scipy in the repo's own toolchain by design):

    uv run --with numpy --with scipy python bench/gpu/k1-calibration.py

Output: bench/results/<date>-gpu-phase2a-k1-calibration.json
CPU-only (numpy/LAPACK) — the hardware-adapter / INVALID rule of
`bench/gpu/README.md` does not apply; no GPU is touched.

@see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md 4.1, 4.2, 5, 6
@see bench/gpu/README.md ("G6 - perEdge conditioning sweep" for the kappa cross-check,
     "Provenance" for the gitShaShort convention)
@see oracle/check_kappa_peredge.py (same assembly path, [G6]'s kappa_2(K))
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from datetime import date

import numpy as np
import scipy.linalg as la

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "oracle")
)
from tpe_constraints_oracle import (  # noqa: E402
    barycenter_block,
    evaluate_constraint_set,
    total_length,
    total_length_block,
)
from tpe_stage1_oracle import (  # noqa: E402
    assemble_inner_product,
    length_weighted_barycenter,
)

ALPHA, BETA, EPS = 3.0, 6.0, 1e-10
U_F32 = 5.96e-8

# The spec's K1 fixture list (64/240/480/960) plus N=120, which K1 does not gate
# but the 4.2 precision-model table does quote — every published row must have
# committed provenance, which was the point of landing this script.  N=64 is also
# the boot self-test's fixture; N=480 is the parent spec's "committed N=480 system".
FIXTURE_N = [64, 120, 240, 480, 960]

# Perturbation directions for the beta bisection.  FIXED and pre-registered so
# a re-run reproduces the published bars bit-for-bit; five is enough to show
# the spread (measured 1.2-1.8x) without hiding the minimum behind a median.
BETA_SEEDS = [0, 1, 2, 3, 4]

# T4 [P 3]: relative residual vs the f64 K, after at most this many refinements.
T4_TOL = 1e-10
T4_MAX_ITERS = 4

# Relative-residual tolerance for the variant-(b) CG probe: the operating tolerance
# an f32 CG inner solve would be run to before f64 IR takes over (spec 3).
CG_TOL = 1e-3

# Bisection bracket for the perturbation scale.  45 halvings of [0, 1e-2] give
# ~3e-16 absolute resolution in s, far below the precision beta is quoted to.
BISECT_HI = 1e-2
BISECT_STEPS = 45


def trefoil(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Same closed-curve parametrization as `src/core/fixtures.ts` `trefoil(n)`
    (verbatim transcription), so the calibrated systems are the milestone's own
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


def assemble(n: int) -> tuple[np.ndarray, np.ndarray]:
    """A (N x N Sobolev inner product) and C (constraint rows) for trefoil(n),
    barycenter + total-length constraint set — the spec's `total` mode.
    @see oracle/check_kappa_peredge.py (identical assembly path)
    """
    V, E = trefoil(n)
    _, _, A, _ = assemble_inner_product(V, E, ALPHA, BETA, EPS)
    blocks = [barycenter_block(length_weighted_barycenter(V, E)), total_length_block(total_length(V, E))]
    _, C, _ = evaluate_constraint_set(blocks, V, E)
    return np.asarray(A, float), np.asarray(C, float)


def make_solver(L, C, Uu, N, k, m, sigma, one):
    """The spec 4.1 shifted-block Cholesky + Schur + Woodbury solve for K, built
    on the (possibly perturbed / f32-derived) factor `L` of A_sigma.  Returns a
    callable r -> z.  This is the numpy twin of what `src/gpu/saddle.ts` will do
    in f64 on the read-back factor.
    @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md 4.1
    """

    def sA(R):
        return la.solve_triangular(L.T, la.solve_triangular(L, R, lower=True), lower=False)

    def sAh(R):
        Rr = R.reshape(3, N, -1) if R.ndim > 1 else R.reshape(3, N, 1)
        o = np.concatenate([sA(Rr[b]) for b in range(3)], 0)
        return o if R.ndim > 1 else o.ravel()

    Z = sAh(C.T)
    S = C @ Z
    # ONE N-backsolve serves all three coordinates: the shift is rank-1 along
    # 1/sqrt(N), identical in each block (spec 4.1 step 3).
    q = sA(one / np.sqrt(N))
    W = np.zeros((m + k, 3))
    for b in range(3):
        yb = np.zeros(m)
        yb[b * N : (b + 1) * N] = q
        lam = np.linalg.solve(S, C @ yb)
        W[:m, b] = yb - Z @ lam
        W[m:, b] = lam
    M = np.eye(3) / sigma - Uu.T @ W

    def solve(r):
        bb, dd = r[:m], r[m:]
        y = sAh(bb)
        lam = np.linalg.solve(S, C @ y - dd)
        zh = np.concatenate([y - Z @ lam, lam])
        return zh + W @ np.linalg.solve(M, Uu.T @ zh)

    return solve


def cg_iters(Amat, b, tol, maxit=20000, m_diag=None) -> int:
    """Unpreconditioned (or Jacobi-preconditioned) CG iterations to `tol` relative
    residual on the shifted SPD block A_sigma — the cost driver of G5's variant (b)
    (f32 GPU CG + f64 IR).  Committed because the whole (b) arm of `G5-confirm`,
    including that gate's own pass/fail thresholds, is priced off this count.
    @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md 3, 5
    """
    x = np.zeros_like(b)
    r = b - Amat @ x
    z = r / m_diag if m_diag is not None else r
    p = z.copy()
    rz = r @ z
    nb = np.linalg.norm(b)
    it = 0
    for it in range(1, maxit + 1):
        Ap = Amat @ p
        a = rz / (p @ Ap)
        x += a * p
        r -= a * Ap
        if np.linalg.norm(r) / nb <= tol:
            break
        z = r / m_diag if m_diag is not None else r
        rz2 = r @ z
        p = z + (rz2 / rz) * p
        rz = rz2
    return it


def calibrate(n: int) -> dict:
    """All four calibrated quantities for one fixture size.  Every number in the
    returned dict is measured here; none is fitted or extrapolated.
    """
    t0 = time.time()
    A, C = assemble(n)
    t_assemble = time.time() - t0

    N = A.shape[0]
    k = C.shape[0]
    m = 3 * N
    one = np.ones(N)
    sigma = float(np.trace(A) / N)
    # Rank-1 shift along the (one-dimensional, spec 2) null direction of A.
    A_sigma = A + (sigma / N) * np.outer(one, one)
    A3 = np.kron(np.eye(3), A)
    K = np.block([[A3, C.T], [C, np.zeros((k, k))]])

    ws = np.linalg.eigvalsh(A_sigma)
    kappa_As = float(ws[-1] / ws[0])
    # K is symmetric indefinite: kappa_2 = max|lambda| / min|lambda|.
    wk = np.abs(np.linalg.eigvalsh(K))
    kappa_K = float(wk.max() / wk.min())

    nAs = np.linalg.norm(A_sigma, "fro")
    Lref = la.cholesky(A_sigma, lower=True)
    L32 = np.linalg.cholesky(A_sigma.astype(np.float32)).astype(float)
    alpha = float(np.linalg.norm(L32 @ L32.T - A_sigma, "fro") / nAs)
    boot_metric = float(np.max(np.abs(L32 - Lref)) / np.max(np.abs(Lref)))
    pivot_floor = float(np.min(np.diag(Lref)))

    P = np.zeros((m, 3))
    for b in range(3):
        P[b * N : (b + 1) * N, b] = 1 / np.sqrt(N)
    Uu = np.vstack([P, np.zeros((k, 3))])

    rng0 = np.random.default_rng(0)
    r_grad = np.concatenate([rng0.standard_normal(m), np.zeros(k)])
    r_proj = np.concatenate([np.zeros(m), rng0.standard_normal(k)])

    def ir_iters(L, r, maxit=T4_MAX_ITERS):
        """Refinements PAST z0 needed to reach T4_TOL, or None if T4 fails.

        The loop must start at z0, not at 0.  The initial approximate solve
        PRODUCES z0 and is not itself a refinement, so seeding `z = 0` spends
        the first pass on the residual of the zero vector -- which is
        identically 1 by construction and can never pass T4_TOL -- and leaves
        only `maxit - 1` real refinements inside the budget.  That is the same
        `z = 0` artefact `ir_curve`'s consumer documents below (`hist[0]`).
        Budget is spec 4.1 step 4: z0 plus T4_MAX_ITERS refinements, restated
        in 4.2 as "z0 + 3 refinements against a budget of z0 + 4".
        Seeding it wrong understates every beta the `betas` bisection below
        derives from this predicate, hence every K1 bar and every e_max.  `_ir_budget_selfcheck` (module level) pins the
        count; run this file with `--self-check` (see `__main__`).
        @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md 4.1, 4.2
        """
        s = make_solver(L, C, Uu, N, k, m, sigma, one)
        nr = np.linalg.norm(r)
        z = s(r)  # z0 -- the initial solve, 0 refinements applied
        for it in range(maxit + 1):  # `it` == refinements applied to z0
            rel = np.linalg.norm(r - K @ z) / nr
            if rel <= T4_TOL:
                return it
            if it == maxit:
                return None
            z = z + s(r - K @ z)
        return None

    def ir_curve(L, r, maxit=8):
        s = make_solver(L, C, Uu, N, k, m, sigma, one)
        nr = np.linalg.norm(r)
        z = np.zeros(m + k)
        hist = []
        for _ in range(maxit + 1):
            rel = float(np.linalg.norm(r - K @ z) / nr)
            hist.append(rel)
            if rel <= T4_TOL:
                break
            z = z + s(r - K @ z)
        return hist

    def bwd(L):
        return float(np.linalg.norm(L @ L.T - A_sigma, "fro") / nAs)

    # beta: bisect the perturbation scale, per direction.  `lo` always passes
    # T4 and `hi` always fails, so `bwd(Lref + lo*E)` is the largest surviving
    # backward error along that direction.
    betas = []
    for sd in BETA_SEEDS:
        rg = np.random.default_rng(sd)
        E = np.tril(rg.standard_normal((N, N)))
        E *= np.linalg.norm(Lref, "fro") / np.linalg.norm(E, "fro")
        lo, hi = 0.0, BISECT_HI
        for _ in range(BISECT_STEPS):
            mid = 0.5 * (lo + hi)
            if ir_iters(Lref + mid * E, r_grad) is not None:
                lo = mid
            else:
                hi = mid
        betas.append(bwd(Lref + lo * E))
    betas = np.array(betas)

    # r0 / rho of the ONE precision model (spec 4.2), driven by the real f32 factor.
    rhs_models = {}
    for name, r in (("gradient", r_grad), ("projection", r_proj)):
        hist = ir_curve(L32, r)
        # hist[0] is the residual of z = 0 and is identically 1 by construction; the
        # spec's r0 is the residual AFTER the first approximate solve, i.e. hist[1],
        # and rho is the contraction it then achieves.
        rhs_models[name] = {
            "r0": hist[1] if len(hist) > 1 else None,
            "rho": (hist[2] / hist[1]) if len(hist) > 2 else None,
            "solvesTo1e10": (len(hist) - 1) if hist[-1] <= T4_TOL else None,
            "residualCurve": hist,
        }

    # Variant (b)'s cost driver, measured rather than bounded (the textbook
    # 0.5*sqrt(kappa)*ln(2/tol) bound is ~4x loose because the spectrum is clustered).
    # The count is RHS-DEPENDENT, so it is measured over the same five directions as
    # beta and reported min/median/max; the spec prices (b) at its MINIMUM, because a
    # faster (b) is the conservative direction for a gate that decides against (a).
    diag_As = np.diag(A_sigma)
    plain = [cg_iters(A_sigma, np.random.default_rng(sd).standard_normal(N), CG_TOL) for sd in BETA_SEEDS]
    jac = [
        cg_iters(A_sigma, np.random.default_rng(sd).standard_normal(N), CG_TOL, m_diag=diag_As)
        for sd in BETA_SEEDS
    ]
    cg = {
        "tol": CG_TOL,
        "seeds": BETA_SEEDS,
        "cgPerSeed": plain,
        "cgMin": min(plain),
        "cgMedian": float(np.median(plain)),
        "cgMax": max(plain),
        "jacobiPcgPerSeed": jac,
        "jacobiPcgMin": min(jac),
        "jacobiPcgMedian": float(np.median(jac)),
        "textbookBound": 0.5 * math.sqrt(kappa_As) * math.log(2 / CG_TOL),
    }

    beta_min = float(betas.min())
    return {
        "n": N,
        "k": k,
        "sigma": sigma,
        "kappaASigma": kappa_As,
        "kappaK": kappa_K,
        "kappaKTimesUf32": kappa_K * U_F32,
        "kappaRatioASigmaOverK": kappa_As / kappa_K,
        "alpha": alpha,
        "betaSeeds": BETA_SEEDS,
        "betaPerSeed": [float(b) for b in betas],
        "betaMin": beta_min,
        "betaMedian": float(np.median(betas)),
        "betaMax": float(betas.max()),
        "betaSpread": float(betas.max() / betas.min()),
        "betaTimesKappaKMin": beta_min * kappa_K,
        "betaTimesKappaKMedian": float(np.median(betas)) * kappa_K,
        # The pre-registered K1 green bar: geometric mean of "indistinguishable
        # from a correct kernel" and "the WORST-CASE fatal level".
        "k1Bar": math.sqrt(alpha * beta_min),
        "k1SlackAboveAlpha": math.sqrt(alpha * beta_min) / alpha,
        "k1SlackBelowBetaMin": beta_min / math.sqrt(alpha * beta_min),
        "bootMetric": boot_metric,
        "minDiagL": pivot_floor,
        "irModel": rhs_models,
        "cgIters": cg,
        "assembleSeconds": t_assemble,
    }


def git_sha_short(rev: str) -> str:
    """Short sha of `rev`.  The spec commit that carries this file is amended
    repeatedly (CLAUDE.md: one commit per spec doc), so the run-time HEAD would
    be keyed to a commit that is about to be rewritten and reachable from no
    branch — the exact failure `bench/gpu/README.md` "Provenance" records.  The
    stable anchor is therefore the spec commit's PARENT.
    @see bench/gpu/README.md ("Provenance")
    """
    return subprocess.run(
        ["git", "rev-parse", "--short", rev], capture_output=True, text=True, check=True
    ).stdout.strip()


def _ir_solve_counts(maxit: int, presolves: int) -> list[tuple[int, int]]:
    """Control-flow twin of `ir_iters`' loop: `(it, solve applications behind the
    vector it checks)` for every convergence check the loop can reach.

    Line-for-line transcription of `ir_iters`, with the vector `z` replaced by a
    count of `s()` applications and the `rel <= T4_TOL` early return dropped, so
    that every check is reached (the worst case the T4 budget is defined over).
    `presolves` is the seeding: 1 for `z = s(r)` (correct, z0), 0 for the
    `z = np.zeros(m + k)` the off-by-one used.
    @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md 4.1, 4.2
    """
    checks, nsolves = [], presolves
    for it in range(maxit + 1):
        checks.append((it, nsolves))  # `rel = norm(r - K @ z) / nr` inspects this z
        if it == maxit:
            break
        nsolves += 1  # `z = z + s(r - K @ z)`
    return checks


def _ir_budget_selfcheck(maxit: int = T4_MAX_ITERS) -> None:
    """Assert `ir_iters` budgets z0 + `maxit` refinements, i.e. its LAST reachable
    convergence check inspects a vector built from `1 + maxit` solve applications.

    Why: this file's bisection predicate silently budgeted one refinement too few
    (it checked the zero vector first), understating every beta and therefore every
    K1 bar and e_max.  A budget is not observable in the JSON, so it gets a test.
    @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md 4.1, 4.2
    """
    fixed = _ir_solve_counts(maxit, presolves=1)
    print(f"ir_iters control-flow twin, T4_MAX_ITERS={maxit}")
    for it, ns in fixed:
        print(f"  it={it}: checks z built from {ns} solve application(s) = z0 + {ns - 1} refinement(s)")
    last_it, last_solves = fixed[-1]
    assert last_solves == 1 + maxit, f"budget is z0 + {last_solves - 1}, spec 4.1 wants z0 + {maxit}"
    assert last_it == maxit, f"returned counter is {last_it} at the last check, want {maxit}"
    assert [it for it, _ in fixed] == [ns - 1 for _, ns in fixed], "`it` must mean refinements applied"

    # The twin discriminates: the old `z = np.zeros(m + k)` seeding is one short.
    buggy_it, buggy_solves = _ir_solve_counts(maxit, presolves=0)[-1]
    assert buggy_solves == maxit, "twin does not reproduce the defect it guards against"
    print(
        f"  PASS: last check = {last_solves} solves = z0 + {last_solves - 1} refinements "
        f"(spec 4.1: z0 + {maxit}).  Old `z = zeros` seeding reached only "
        f"{buggy_solves} solves = z0 + {buggy_solves - 1}, and mislabelled it as it={buggy_it}."
    )


def main() -> None:
    rows = [calibrate(n) for n in FIXTURE_N]
    for r in rows:
        print(
            f"N={r['n']:4d} kappa(K)={r['kappaK']:.4e} alpha={r['alpha']:.3e} "
            f"beta[min/med/max]={r['betaMin']:.3e}/{r['betaMedian']:.3e}/{r['betaMax']:.3e} "
            f"({r['betaSpread']:.2f}x) K1bar=sqrt(alpha*betaMin)={r['k1Bar']:.3e} "
            f"boot={r['bootMetric']:.3e} "
            f"CG[min/med/max]={r['cgIters']['cgMin']}/{r['cgIters']['cgMedian']:.0f}/"
            f"{r['cgIters']['cgMax']} PCGmin={r['cgIters']['jacobiPcgMin']}"
        )
    out = {
        "gate": "phase2aK1Calibration",
        "gitShaShort": git_sha_short("HEAD~1"),
        "gitShaNote": (
            "parent of the (amended) Phase 2 spec commit that adds this script; "
            "the measured inputs (oracle/, src/core/fixtures.ts) are unchanged from that tree"
        ),
        "date": date.today().isoformat(),
        "tool": f"numpy {np.__version__} / scipy {sp_version()}",
        "constraintMode": "total",
        "fixture": "trefoil",
        "uF32": U_F32,
        "t4": {"tol": T4_TOL, "maxIters": T4_MAX_ITERS},
        "betaSeeds": BETA_SEEDS,
        "rows": rows,
    }
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "results",
        f"{out['date']}-gpu-phase2a-k1-calibration.json",
    )
    with open(out_path, "w") as f:
        f.write(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("wrote", os.path.normpath(out_path))


def sp_version() -> str:
    import scipy

    return scipy.__version__


if __name__ == "__main__":
    # `--self-check` runs the IR-budget twin only (seconds); a bare run does the
    # full multi-minute calibration.  Same invocation either way:
    #   uv run --with numpy --with scipy python bench/gpu/k1-calibration.py [--self-check]
    # @see `_ir_budget_selfcheck`
    if "--self-check" in sys.argv:
        _ir_budget_selfcheck()
    else:
        main()
