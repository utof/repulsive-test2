/**
 * Mass-lumped L²ₕ curve norm, constraint-set projection, and the backtracking
 * line search for the fractional Sobolev descent step (Repulsive Curves,
 * Yu/Schumacher/Crane 2021). Projection and line search are generic over a
 * stacked {@link ConstraintSet} (spec §3.1); the original barycenter-only
 * entry points survive as bit-identical delegating wrappers (spec §3.2).
 *
 * IMPORTANT: the discrete L²ₕ norm definition and every constant in this module
 * (c₁, ρ, τ₀, τ_min, the projection tolerances and max_iter) are OUR tunable
 * choices, not paper constants — the paper never defines its discrete L² norm
 * or its backtracking constants. Do not treat them as paper ground truth when
 * comparing against other implementations.
 * @see local_files/2026-07-02-sobolev-formula-audit.md (Item 9 — "Unstated inventions")
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Constraint projection after a step"), §C
 * @see oracle/tpe_stage1_oracle.py (l2_curve_norm_vec3 / barycenter_scale / project_barycenter / line_search_step)
 * @see oracle/tpe_constraints_oracle.py (project_constraint_set / line_search_step_set)
 */

import { calculateEnergy } from '../tangentPointEnergy';
import type { Edge, Vec3 } from '../testConfigs';
import { barycenterBlock, type ConstraintSet } from './constraintSet';
import { assembleA } from './innerProduct';
import { expandBlockDiag, flatten, unflatten } from './layout';
import { solveSaddle } from './linsolve';

/**
 * Mass-lumped discrete curve L²ₕ norm of a per-vertex Vec3 field:
 * ‖v‖²_{L²ₕ} = Σ_{I=(i1,i2)} (ℓ_I/2)·(|v_{i1}|² + |v_{i2}|²).
 *
 * Uses RAW geometric lengths ℓ_I = ‖e_I‖ (no +ε) — the norm is geometric, not
 * part of the regularized energy; same convention as the constraint machinery
 * in `./constraints` (do NOT "unify" with the ℓ^ε of innerProduct.ts).
 * This mass-lumped definition itself is OUR choice, not the paper's — see the
 * module-header caution.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C ("mass-lumped discrete curve L² norm")
 * @see local_files/2026-07-02-sobolev-formula-audit.md (Item 9)
 * @see oracle/tpe_stage1_oracle.py (l2_curve_norm_vec3)
 */
export function l2CurveNorm(values: Vec3[], vertices: Vec3[], edges: Edge[]): number {
    let total = 0;
    for (const [i1, i2] of edges) {
        const p1 = vertices[i1];
        const p2 = vertices[i2];
        const ex = p2[0] - p1[0];
        const ey = p2[1] - p1[1];
        const ez = p2[2] - p1[2];
        const ell = Math.sqrt(ex * ex + ey * ey + ez * ez);
        const v1 = values[i1];
        const v2 = values[i2];
        total +=
            0.5 *
            ell *
            (v1[0] * v1[0] +
                v1[1] * v1[1] +
                v1[2] * v1[2] +
                (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]));
    }
    // max(0, ·) guards roundoff-negative accumulation before the sqrt, mirroring
    // the oracle's math.sqrt(max(0.0, total)).
    return Math.sqrt(Math.max(0, total));
}

/**
 * Scale factor for the projection stopping tolerance: max(1, L·max(1, R)) with
 * L = Σ_I ℓ_I (raw lengths) and R = max over edge endpoints of ‖p − x₀‖.
 * Feeds the relative tolerance max(tolAbs, tolRel·scale) via the barycenter
 * block's `scale` (spec §3.3) — this scaling is OUR choice, not the paper's
 * (module-header caution).
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Constraint projection after a step" — stopping tolerance)
 * @see oracle/tpe_stage1_oracle.py (barycenter_scale)
 */
export function barycenterScale(vertices: Vec3[], edges: Edge[], x0: Vec3): number {
    let L = 0;
    let R = 0;
    for (const [i1, i2] of edges) {
        const p1 = vertices[i1];
        const p2 = vertices[i2];
        const ex = p2[0] - p1[0];
        const ey = p2[1] - p1[1];
        const ez = p2[2] - p1[2];
        L += Math.sqrt(ex * ex + ey * ey + ez * ez);
        const d1x = p1[0] - x0[0];
        const d1y = p1[1] - x0[1];
        const d1z = p1[2] - x0[2];
        const d2x = p2[0] - x0[0];
        const d2y = p2[1] - x0[1];
        const d2z = p2[2] - x0[2];
        R = Math.max(
            R,
            Math.sqrt(d1x * d1x + d1y * d1y + d1z * d1z),
            Math.sqrt(d2x * d2x + d2y * d2y + d2z * d2z),
        );
    }
    return Math.max(1, L * Math.max(1, R));
}

/**
 * Options for {@link projectOntoConstraintSet}. Defaults (tolAbs = tolRel =
 * 1e-10, maxIter = 8) are OUR tunables, not paper constants — see module
 * header.
 * @see local_files/2026-07-02-sobolev-formula-audit.md (Item 7 — "Flags")
 */
export interface ProjectConstraintSetOptions {
    tolAbs?: number;
    tolRel?: number;
    maxIter?: number;
}

/**
 * Back-compat alias (spec §3.2) — the pre-ConstraintSet name of
 * {@link ProjectConstraintSetOptions}, structurally identical.
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.2
 */
export type ProjectBarycenterOptions = ProjectConstraintSetOptions;

/**
 * Result of {@link projectOntoConstraintSet}. `vertices` is the final iterate
 * even on failure; `phiNorm` is the STACKED ‖Φ‖₂ across all blocks.
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.1, §3.3
 */
export interface ProjectConstraintSetResult {
    vertices: Vec3[];
    ok: boolean;
    iterations: number;
    phiNorm: number;
}

/**
 * Back-compat alias (spec §3.2) — the pre-ConstraintSet name of
 * {@link ProjectConstraintSetResult}, structurally identical.
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.2
 */
export type ProjectBarycenterResult = ProjectConstraintSetResult;

/**
 * Projects a candidate curve back onto the stacked constraints Φ(γ) = 0 by
 * repeated Newton-like corrections: at each iterate solve
 * `[[Ā, Cᵀ], [C, 0]]·[x; μ] = [0; −Φ(γ^q)]` (Φ, C stacked over all blocks in
 * set order) and update γ^{q+1} = γ^q + x, until EVERY block b satisfies
 * ‖Φ_b‖₂ ≤ max(tolAbs, tolRel·b.scale(γ^q)) — the per-block stopping rule of
 * spec §3.3 (OUR invention, not paper-sourced; reduces exactly to the original
 * barycenter rule for the singleton barycenter set). The EMPTY set converges
 * trivially at iteration 0 (spec §9a).
 *
 * Ā and C are REASSEMBLED at every projection iterate — a results-doc choice;
 * the paper leaves freeze-vs-reassemble during projection unspecified. Do NOT
 * "optimize" to frozen matrices without a measured decision.
 * @see local_files/2026-07-02-sobolev-formula-audit.md (Item 7 — reassembly not in the paper)
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Constraint projection after a step")
 *
 * A non-finite correction step or a solver throw returns `ok: false` with the
 * current iterate, mirroring the oracle's failure semantics.
 * @see oracle/tpe_constraints_oracle.py (project_constraint_set)
 */
export function projectOntoConstraintSet(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
    set: ConstraintSet,
    opts?: ProjectConstraintSetOptions,
): ProjectConstraintSetResult {
    const tolAbs = opts?.tolAbs ?? 1e-10;
    const tolRel = opts?.tolRel ?? 1e-10;
    const maxIter = opts?.maxIter ?? 8;

    let cur: Vec3[] = vertices.map((p) => [p[0], p[1], p[2]]);
    let finalPhiNorm = Number.POSITIVE_INFINITY;
    // maxIter + 1 convergence CHECKS (iterations 0..maxIter): the oracle's
    // range(max_iter + 1) loop checks once before any correction, so a
    // candidate already on the constraints converges with iterations = 0.
    // @see oracle/tpe_stage1_oracle.py (project_barycenter)
    for (let it = 0; it <= maxIter; it++) {
        // Evaluate each block ONCE per iterate; the per-block slices feed the
        // §3.3 convergence rule, their concatenation feeds the saddle solve.
        const evals = set.map((block) => block.evaluate(cur, edges));
        const phi: number[] = [];
        const C: number[][] = [];
        for (const e of evals) {
            phi.push(...e.phi);
            C.push(...e.C);
        }
        let phiNormSq = 0;
        for (const v of phi) phiNormSq += v * v;
        finalPhiNorm = Math.sqrt(phiNormSq);

        // Per-block §3.3 check: converged iff EVERY block is within its own
        // scaled tolerance (vacuously true for the empty set).
        let converged = true;
        for (let b = 0; b < set.length; b++) {
            let sq = 0;
            for (const v of evals[b].phi) sq += v * v;
            const blockNorm = Math.sqrt(sq);
            if (blockNorm > Math.max(tolAbs, tolRel * set[b].scale(cur, edges))) {
                converged = false;
                break;
            }
        }
        if (converged) {
            return { vertices: cur, ok: true, iterations: it, phiNorm: finalPhiNorm };
        }
        if (it === maxIter) {
            break;
        }
        try {
            // Reassemble Ā(current iterate) — see the anchor in the TSDoc above.
            const A = assembleA(cur, edges, disjointPairs, alpha, beta, epsilon);
            const A3 = expandBlockDiag(A);
            const negPhi = phi.map((v) => -v);
            const { x } = solveSaddle(A3, C, new Array<number>(3 * cur.length).fill(0), negPhi);
            const step = unflatten(x);
            for (const s of step) {
                if (!Number.isFinite(s[0]) || !Number.isFinite(s[1]) || !Number.isFinite(s[2])) {
                    return { vertices: cur, ok: false, iterations: it, phiNorm: finalPhiNorm };
                }
            }
            cur = cur.map((p, i) => [p[0] + step[i][0], p[1] + step[i][1], p[2] + step[i][2]]);
        } catch {
            return { vertices: cur, ok: false, iterations: it, phiNorm: finalPhiNorm };
        }
    }
    return { vertices: cur, ok: false, iterations: maxIter, phiNorm: finalPhiNorm };
}

/**
 * Back-compat x₀ signature: delegates to {@link projectOntoConstraintSet} with
 * the barycenter-only set — numerically bit-identical to the pre-ConstraintSet
 * implementation (regression-proven by the unmodified stage-1 golden tests,
 * spec §3.2).
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.2
 * @see oracle/tpe_stage1_oracle.py (project_barycenter)
 */
export function projectBarycenter(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
    x0: Vec3,
    opts?: ProjectBarycenterOptions,
): ProjectBarycenterResult {
    return projectOntoConstraintSet(
        vertices,
        edges,
        disjointPairs,
        alpha,
        beta,
        epsilon,
        [barycenterBlock(x0)],
        opts,
    );
}

/**
 * Options for {@link lineSearchStepSet}. Defaults (c1 = 1e-4, shrink ρ = 1/2,
 * tau0 = 1, tauMin = 1e-12) are OUR tunables — the paper only says "backtracking
 * line search starting with τ=1" and cites Boyd Alg. 9.2 without constants.
 * @see local_files/2026-07-02-sobolev-formula-audit.md (Item 9 — "Unstated inventions")
 */
export interface LineSearchOptions {
    c1?: number;
    shrink?: number;
    tau0?: number;
    tauMin?: number;
}

/** Failure reasons of {@link lineSearchStepSet}, mirroring the oracle's `reason` strings. */
export type LineSearchFailureReason =
    | 'zero_or_nonfinite_gradient_norm'
    | 'not_a_descent_direction'
    | 'armijo_failed'
    | 'projection_failed'
    | 'tau_below_min';

/**
 * Result of {@link lineSearchStepSet}. On failure `vertices` echoes the input
 * (unchanged), `tau` is 0, `energyAfter` equals `energyBefore`, and the
 * projection fields are null — mirroring the oracle's failure payloads.
 * `slope`/`gradientL2Norm` are absent only on branches where the oracle never
 * computed them.
 * @see oracle/tpe_stage1_oracle.py (line_search_step)
 */
export interface LineSearchStepResult {
    accepted: boolean;
    reason?: LineSearchFailureReason;
    tau: number;
    energyBefore: number;
    energyAfter: number;
    slope?: number;
    gradientL2Norm?: number;
    vertices: Vec3[];
    projectionIterations: number | null;
    projectionPhiNorm: number | null;
}

/**
 * One backtracking line-search step of the constrained Sobolev descent flow
 * (spec §C): normalize g̃ in the L²ₕ norm, walk along −p (the DESCENT direction
 * is minus the normalized gradient — the solve returns the ascent
 * representative, spec §C step 7 "The step direction is (−p)"), project each
 * trial point onto the stacked {@link ConstraintSet}, and accept the first τ
 * in {τ₀, τ₀ρ, τ₀ρ², …, ≥ τ_min} satisfying Armijo
 * E(γ_proj) ≤ E₀ − c₁·τ·m with slope m = dEᵀp.
 *
 * `dE` is a PARAMETER (same pluggability contract as
 * `solveConstrainedGradientSet` in `./gradient`); the ENERGY is not — it is
 * the app's own `calculateEnergy`, because Armijo must gate on the exact
 * energy the flow is minimizing. The set must be the SAME frozen-target set
 * used for the gradient solve (spec §3.5).
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C
 * @see local_files/sobolev-gradient-handoff.md §1 ("The dE fed into the solve must be pluggable")
 * @see oracle/tpe_constraints_oracle.py (line_search_step_set)
 */
export function lineSearchStepSet(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
    dE: Vec3[],
    gTilde: Vec3[],
    set: ConstraintSet,
    opts?: LineSearchOptions,
): LineSearchStepResult {
    const c1 = opts?.c1 ?? 1e-4;
    const shrink = opts?.shrink ?? 0.5;
    const tau0 = opts?.tau0 ?? 1;
    const tauMin = opts?.tauMin ?? 1e-12;

    const e0 = calculateEnergy(vertices, edges, disjointPairs, alpha, beta, epsilon);
    const gradNorm = l2CurveNorm(gTilde, vertices, edges);
    if (!Number.isFinite(gradNorm) || gradNorm <= 0) {
        return {
            accepted: false,
            reason: 'zero_or_nonfinite_gradient_norm',
            tau: 0,
            energyBefore: e0,
            energyAfter: e0,
            vertices,
            projectionIterations: 0,
            projectionPhiNorm: null,
        };
    }

    const direction: Vec3[] = gTilde.map((g) => [
        g[0] / gradNorm,
        g[1] / gradNorm,
        g[2] / gradNorm,
    ]);
    const dEFlat = flatten(dE);
    const dirFlat = flatten(direction);
    let slope = 0;
    for (let i = 0; i < dEFlat.length; i++) {
        slope += dEFlat[i] * dirFlat[i];
    }
    if (!Number.isFinite(slope) || slope <= 0) {
        return {
            accepted: false,
            reason: 'not_a_descent_direction',
            tau: 0,
            energyBefore: e0,
            energyAfter: e0,
            slope,
            vertices,
            projectionIterations: 0,
            projectionPhiNorm: null,
        };
    }

    let tau = tau0;
    let lastReason: LineSearchFailureReason = 'tau_below_min';
    while (tau >= tauMin) {
        // Candidate = γ − τ·p: MINUS, because p is the normalized (ascent)
        // gradient and the descent direction is −p — spec §C steps 7 and 9.
        // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C
        const raw: Vec3[] = vertices.map((p, i) => [
            p[0] - tau * direction[i][0],
            p[1] - tau * direction[i][1],
            p[2] - tau * direction[i][2],
        ]);
        const proj = projectOntoConstraintSet(raw, edges, disjointPairs, alpha, beta, epsilon, set);
        const projFinite = proj.vertices.every(
            (p) => Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2]),
        );
        if (proj.ok && projFinite) {
            const e1 = calculateEnergy(proj.vertices, edges, disjointPairs, alpha, beta, epsilon);
            if (Number.isFinite(e1) && e1 <= e0 - c1 * tau * slope) {
                return {
                    accepted: true,
                    tau,
                    energyBefore: e0,
                    energyAfter: e1,
                    slope,
                    gradientL2Norm: gradNorm,
                    vertices: proj.vertices,
                    projectionIterations: proj.iterations,
                    projectionPhiNorm: proj.phiNorm,
                };
            }
            lastReason = 'armijo_failed';
        } else {
            lastReason = 'projection_failed';
        }
        tau *= shrink;
    }

    return {
        accepted: false,
        reason: lastReason,
        tau: 0,
        energyBefore: e0,
        energyAfter: e0,
        slope,
        gradientL2Norm: gradNorm,
        vertices,
        projectionIterations: null,
        projectionPhiNorm: null,
    };
}

/**
 * Back-compat x₀ signature: delegates to {@link lineSearchStepSet} with the
 * barycenter-only set — numerically bit-identical to the pre-ConstraintSet
 * implementation (regression-proven by the unmodified stage-1 golden tests,
 * spec §3.2).
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.2
 * @see oracle/tpe_stage1_oracle.py (line_search_step)
 */
export function lineSearchStep(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    alpha: number,
    beta: number,
    epsilon: number,
    dE: Vec3[],
    gTilde: Vec3[],
    x0: Vec3,
    opts?: LineSearchOptions,
): LineSearchStepResult {
    return lineSearchStepSet(
        vertices,
        edges,
        disjointPairs,
        alpha,
        beta,
        epsilon,
        dE,
        gTilde,
        [barycenterBlock(x0)],
        opts,
    );
}
