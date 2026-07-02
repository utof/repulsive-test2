import { solveConstrainedGradient } from './sobolev/gradient';
import { type LineSearchFailureReason, l2CurveNorm, lineSearchStep } from './sobolev/lineSearch';
import { calculateEnergy, gradientAnalytical, gradientFiniteDiff } from './tangentPointEnergy';
import type { Edge, Vec3 } from './testConfigs';

// Physical/numeric constants for the tangent-point descent. These were hardcoded
// in the old src/index.tsx; centralised here so the store and scene share one source.
// @see docs/superpowers/specs/2026-07-02-react-three-webgpu-switch-design.md §4.1
export const DEFAULTS = { alpha: 3, beta: 6, epsilon: 1e-10, h: 1e-6 } as const;

export interface StepOptions {
    mode: 'analytical' | 'finiteDiff';
    stepSize: number;
    alpha?: number;
    beta?: number;
    epsilon?: number;
    h?: number;
}

// One gradient-descent step. Pure: returns NEW arrays, never mutates inputs.
// Mirrors the old animate() sequence exactly: grad -> v - stepSize*grad -> energy(new v).
export function step(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    opts: StepOptions,
): { vertices: Vec3[]; energy: number } {
    const alpha = opts.alpha ?? DEFAULTS.alpha;
    const beta = opts.beta ?? DEFAULTS.beta;
    const epsilon = opts.epsilon ?? DEFAULTS.epsilon;
    const h = opts.h ?? DEFAULTS.h;

    const grad =
        opts.mode === 'analytical'
            ? gradientAnalytical(vertices, edges, disjointPairs, alpha, beta, epsilon)
            : gradientFiniteDiff(vertices, edges, disjointPairs, alpha, beta, epsilon, h);

    const next: Vec3[] = vertices.map((v, i) => [
        v[0] - opts.stepSize * grad[i][0],
        v[1] - opts.stepSize * grad[i][1],
        v[2] - opts.stepSize * grad[i][2],
    ]);

    const energy = calculateEnergy(next, edges, disjointPairs, alpha, beta, epsilon);
    return { vertices: next, energy };
}

/**
 * Termination threshold for the Sobolev descent: stop when ‖g̃‖_{L²ₕ} < 1e-4.
 * The 1e-4 value IS paper-sourced ("In our examples we use ε = 10⁻⁴"), unlike
 * the line-search constants and the mass-lumped norm definition, which are our
 * tunables — do not lump this constant in with those when re-tuning.
 * @see local_files/2026-07-02-sobolev-formula-audit.md (Item 9 — paper-sourced vs unstated inventions)
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C (step 5)
 */
export const SOBOLEV_CONVERGENCE_TOL = 1e-4;

/**
 * Options for {@link sobolevStep}. `mode` selects how the differential dE is
 * produced (same pluggability contract as {@link step} and
 * `solveConstrainedGradient`); the solve/line-search behavior is identical
 * either way.
 * @see local_files/sobolev-gradient-handoff.md §1 ("The dE fed into the solve must be pluggable")
 */
export interface SobolevStepOptions {
    mode: 'analytical' | 'finiteDiff';
    alpha?: number;
    beta?: number;
    epsilon?: number;
    h?: number;
}

/**
 * Rejection reasons of {@link sobolevStep}: the line search's own reasons plus
 * 'singular_system' — the saddle solve threw (exactly singular Ā, e.g. an
 * isolated vertex whose Ā rows / C columns are all zero). The solve failure is
 * folded into the same reject-report contract as line-search failure so the
 * frame loop never sees an exception.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C (step 10)
 * @see src/core/sobolev/linsolve.ts (luSolve singularity throw)
 */
export type SobolevStepFailureReason = LineSearchFailureReason | 'singular_system';

/**
 * Per-step diagnostics of {@link sobolevStep}, surfaced to the UI. `tau` is 0
 * when no step was taken (converged or rejected); `projectionIterations` is
 * null when the line search failed before/without a successful projection;
 * `reason` is present only on a rejected step. On 'singular_system' the
 * residual/gradientL2Norm are NaN — the solve never produced them.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C (steps 5, 10)
 */
export interface SobolevStepStats {
    tau: number;
    residual: number;
    gradientL2Norm: number;
    projectionIterations: number | null;
    reason?: SobolevStepFailureReason;
}

/**
 * One step of the constrained fractional Sobolev descent (Repulsive Curves,
 * Yu/Schumacher/Crane 2021), spec §C: dE → constrained saddle solve for g̃ →
 * termination check → backtracking line search with barycenter projection.
 * Pure like {@link step}: returns NEW arrays, never mutates inputs.
 *
 * Outcomes:
 * - converged (‖g̃‖_{L²ₕ} < {@link SOBOLEV_CONVERGENCE_TOL}, spec §C step 5):
 *   input vertices echoed unchanged, `converged: true`, `accepted: false`.
 * - rejected line search (spec §C step 10): input vertices echoed unchanged,
 *   `accepted: false`, `stats.reason` set — reject/leave-unchanged/report,
 *   never throw.
 * - accepted: projected new vertices and their energy.
 *
 * `x0` is the FROZEN barycenter target (computed once per run start via
 * `barycenterTarget`, never from the current iterate — see the x0 contract on
 * `solveConstrainedGradient`).
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C
 * @see oracle/tpe_stage1_oracle.py (solve_constrained_gradient / line_search_step)
 */
export function sobolevStep(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    x0: Vec3,
    opts: SobolevStepOptions,
): {
    vertices: Vec3[];
    energy: number;
    accepted: boolean;
    converged: boolean;
    stats: SobolevStepStats;
} {
    const alpha = opts.alpha ?? DEFAULTS.alpha;
    const beta = opts.beta ?? DEFAULTS.beta;
    const epsilon = opts.epsilon ?? DEFAULTS.epsilon;
    const h = opts.h ?? DEFAULTS.h;

    const dE =
        opts.mode === 'analytical'
            ? gradientAnalytical(vertices, edges, disjointPairs, alpha, beta, epsilon)
            : gradientFiniteDiff(vertices, edges, disjointPairs, alpha, beta, epsilon, h);

    let gTilde: Vec3[];
    let residual: number;
    try {
        ({ gTilde, residual } = solveConstrainedGradient(
            vertices,
            edges,
            disjointPairs,
            alpha,
            beta,
            epsilon,
            dE,
            x0,
        ));
    } catch {
        // Exactly singular saddle system (e.g. an isolated vertex → zero Ā rows).
        // Fold into the spec §C step 10 contract: reject, echo the input, report —
        // the frame loop auto-pauses on accepted:false instead of crashing.
        // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C (step 10)
        return {
            vertices,
            energy: calculateEnergy(vertices, edges, disjointPairs, alpha, beta, epsilon),
            accepted: false,
            converged: false,
            stats: {
                tau: 0,
                residual: Number.NaN,
                gradientL2Norm: Number.NaN,
                projectionIterations: null,
                reason: 'singular_system',
            },
        };
    }

    const gradientL2Norm = l2CurveNorm(gTilde, vertices, edges);
    if (gradientL2Norm < SOBOLEV_CONVERGENCE_TOL) {
        return {
            vertices,
            energy: calculateEnergy(vertices, edges, disjointPairs, alpha, beta, epsilon),
            accepted: false,
            converged: true,
            stats: { tau: 0, residual, gradientL2Norm, projectionIterations: 0 },
        };
    }

    const result = lineSearchStep(
        vertices,
        edges,
        disjointPairs,
        alpha,
        beta,
        epsilon,
        dE,
        gTilde,
        x0,
    );
    // On failure lineSearchStep already echoes the input vertices unchanged and
    // reports energyAfter = energyBefore — spec §C step 10 semantics pass through.
    return {
        vertices: result.vertices,
        energy: result.energyAfter,
        accepted: result.accepted,
        converged: false,
        stats: {
            tau: result.tau,
            residual,
            gradientL2Norm,
            projectionIterations: result.projectionIterations,
            ...(result.reason !== undefined ? { reason: result.reason } : {}),
        },
    };
}
