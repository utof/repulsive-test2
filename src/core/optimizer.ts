import { barycenterBlock, type ConstraintSet } from './sobolev/constraintSet';
import { solveConstrainedGradientSetFrozen } from './sobolev/gradient';
import { type LineSearchFailureReason, l2CurveNorm, lineSearchStepSet } from './sobolev/lineSearch';
import type { FactorMode, FrozenSaddleOperator } from './sobolev/linsolve';
import {
    type PenaltyConfig,
    penaltiesActive,
    penaltyEnergy,
    penaltyGradient,
} from './sobolev/penalties';
import { type SobolevStepTimings, timed, timingsBegin, timingsEnd } from './sobolev/phaseTimings';
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
    /**
     * Opt into returning `field` = the raw L² gradient dE this step took (the same
     * array computed internally and otherwise discarded), for the GradientArrows
     * diagnostic (§D14 / issue #9). Absent ⇒ `field` omitted, every path
     * bit-identical to today (no extra allocation — `grad` is computed regardless;
     * this only decides whether the reference rides out on the return object).
     * @see docs/superpowers/plans/2026-07-04-worker-solver.md §D14
     */
    collectField?: boolean;
}

// One gradient-descent step. Pure: returns NEW arrays, never mutates inputs.
// Mirrors the old animate() sequence exactly: grad -> v - stepSize*grad -> energy(new v).
export function step(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    opts: StepOptions,
): { vertices: Vec3[]; energy: number; field?: Vec3[] } {
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
    // §D14: surface the raw dE (at the INPUT vertices) ONLY when asked — absent key
    // otherwise so the return shape stays bit-identical for every existing caller.
    // @see docs/superpowers/plans/2026-07-04-worker-solver.md §D14
    if (opts.collectField) return { vertices: next, energy, field: grad };
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
 * Constraint-projection solve strategy of the Sobolev step (solver-perf
 * Task 6) — see the `projectionMode` option on {@link SobolevStepOptions} for
 * the semantics and the measured trade-off.
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 6)
 */
export type ProjectionMode = 'reassemble' | 'frozen';

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
    /**
     * Opt into per-phase wall-clock collection (default false). When absent the
     * phase wraps are provably inert (see {@link timed}) and `sobolevStepSet`'s
     * outputs are bit-identical to today — the golden suites are the backstop.
     * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 1)
     */
    collectTimings?: boolean;
    /**
     * Opt into returning `descentField` = the constrained Sobolev gradient g̃ this
     * step solved for (over the FULL ConstraintSet), for the GradientArrows
     * diagnostic (§D14 / issue #9). Absent ⇒ `descentField` omitted, every path
     * bit-identical to today (g̃ is computed regardless; this only decides whether
     * the reference is exposed). `null` on a singular saddle (no g̃ was produced).
     * @see docs/superpowers/plans/2026-07-04-worker-solver.md §D14
     */
    collectField?: boolean;
    /**
     * Precomputed E₀ at the CURRENT input vertices, forwarded to the line
     * search (and reused on the converged/singular echo paths, which return
     * these same vertices) instead of recomputing. MUST be exactly the
     * OBJECTIVE at γ₀: `calculateEnergy(vertices, edges, disjointPairs,
     * alpha, beta, epsilon)` plus `penaltyEnergy(vertices, edges, penalties)`
     * when `penalties` is active — a continuous run gets this for free from
     * the previous accepted step's returned energy UNDER THE SAME penalty
     * config (a config change invalidates the cache; the caller owns that
     * too). A stale value corrupts the Armijo gate; the caller owns the
     * invariant (same contract as the line search's `energyBefore` option in
     * `./sobolev/lineSearch`). Omitted → E₀ recomputed (unchanged behavior).
     * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 4)
     * @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2.4
     */
    energyBefore?: number;
    /**
     * Soft-constraint penalties (5C): the paper's catalog (total length,
     * length difference, field alignment) entering the OBJECTIVE — analytic
     * penalty gradients ADD to dE before the saddle solve, penalty energies
     * ADD to the Armijo gate and to the returned `energy`. Never constraint
     * rows; H^s inner product unchanged (SelfAvoiding.tex line 769). Absent
     * or all-zero ⇒ every code path bit-identical to the penalty-free build.
     * @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md
     * @see src/core/sobolev/penalties.ts
     */
    penalties?: PenaltyConfig;
    /**
     * Constraint-projection solve strategy (solver-perf Task 6).
     * 'frozen': one K(γ₀) LU per step serves the gradient solve AND every
     * projection iterate of every τ-trial (fresh −Φ RHS each iterate) — the
     * authors' reference-implementation scheme (ythea/repulsive-curves
     * src/tpe_flow_sc.cpp; paper line 734); cheaper (no per-iterate
     * reassembly + refactorization) but a stale-Jacobian quasi-Newton
     * projection: on junction-heavy fixtures the τ=1 trial can fail to
     * project and the step backtracks (measured: junction-y-edgelengths
     * τ 1.0 → 0.5 vs 'reassemble' — oracle/README.md table).
     * 'reassemble' (default, and the semantics of ALL pre-existing goldens):
     * rebuild + refactor Ā/C at every projection iterate — stricter step
     * quality, more dense work.
     * @see oracle/README.md ("Frozen-projection mode")
     * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 6)
     */
    projectionMode?: ProjectionMode;
    /**
     * Dense factorization for EVERY saddle solve of the step (gradient solve
     * + all projection iterates, frozen or reassembled — LDLᵀ A/B).
     * 'ldlt' (default since 2026-07-06, when the pre-registered gates passed
     * — factor-phase p50 ≥1.44× over LU on every bench case, all
     * golden-gated tests green): Bunch–Kaufman symmetric-indefinite LDLᵀ,
     * ~half the factor flops. 'lu': dense LU with partial pivoting — the A/B
     * leg, bit-identical to the pre-2026-07-06 build. Both paths are gated
     * by the same self-certifying residual, so this is a cost knob, not a
     * semantics knob (which is why there is no store/UI toggle — unlike
     * projectionMode).
     * @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (pinned decision 4 + verdict)
     */
    factorMode?: FactorMode;
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
 * Yu/Schumacher/Crane 2021), spec §C, generic over a stacked
 * {@link ConstraintSet} (spec §3.1; the EMPTY set = unconstrained Sobolev
 * flow, spec §9a): dE → constrained saddle solve for g̃ → termination check →
 * backtracking line search with constraint-set projection.
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
 * The set's TARGETS (x₀, L⁰, …) are FROZEN at block construction (once per
 * run start, spec §3.5) — pass the same set every step; never rebuild blocks
 * from the current iterate. Programmatic composition should be validated once
 * with `assertValidConstraintSet` at construction time (spec §3.4) — this
 * function deliberately does NOT validate per step, because it runs in the
 * frame loop and must never throw (invalid rank surfaces as the existing
 * 'singular_system' rejection instead).
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.1, §3.4, §3.5
 * @see oracle/tpe_constraints_oracle.py (solve_constrained_gradient_set / line_search_step_set)
 */
export function sobolevStepSet(
    vertices: Vec3[],
    edges: Edge[],
    disjointPairs: number[][],
    set: ConstraintSet,
    opts: SobolevStepOptions,
): {
    vertices: Vec3[];
    energy: number;
    accepted: boolean;
    converged: boolean;
    stats: SobolevStepStats;
    timings?: SobolevStepTimings;
    // §D14: the g̃ this step solved for (present only when collectField was set;
    // null when the saddle was singular so no g̃ exists). @see plan §D14 / issue #9
    descentField?: Vec3[] | null;
} {
    // Phase-timing collection is opt-in and provably inert when off: timingsBegin
    // arms the module collector, timed('step', …) records the whole step, and the
    // inner call-site wraps (dE / energy / lineSearch here, plus the assembleA /
    // saddle / projection wraps deeper in the pipeline) attach to the same ledger.
    // With collectTimings absent, timed() is a plain call and the result is
    // bit-identical to today (the golden suites are the backstop).
    // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 1)
    const collect = opts.collectTimings ?? false;
    // §D14: capture the g̃ the step solves for (raw reference — zero alloc) so it can
    // ride out on the return like `timings`. Stays null on the singular-saddle path
    // (no g̃), and is exposed only when collectField was requested (else the return
    // shape is bit-identical to today). @see plan §D14 / issue #9.
    const collectField = opts.collectField ?? false;
    let capturedField: Vec3[] | null = null;
    if (collect) timingsBegin();
    const outcome = timed(
        'step',
        (): {
            vertices: Vec3[];
            energy: number;
            accepted: boolean;
            converged: boolean;
            stats: SobolevStepStats;
        } => {
            const alpha = opts.alpha ?? DEFAULTS.alpha;
            const beta = opts.beta ?? DEFAULTS.beta;
            const epsilon = opts.epsilon ?? DEFAULTS.epsilon;
            const h = opts.h ?? DEFAULTS.h;
            const pen = penaltiesActive(opts.penalties) ? opts.penalties : undefined;

            const dETpe = timed('dE', () =>
                opts.mode === 'analytical'
                    ? gradientAnalytical(vertices, edges, disjointPairs, alpha, beta, epsilon)
                    : gradientFiniteDiff(vertices, edges, disjointPairs, alpha, beta, epsilon, h),
            );
            // Penalties enter the objective's differential BEFORE the solve
            // (plan §2.4): dE_total = dE_tpe + Σ w·dÊ. Inactive ⇒ dE IS dETpe
            // (same array — bit-identical path). Outside the 'dE' timing wrap:
            // penaltyGradient is O(E), negligible next to the O(E²) kernel.
            // @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2.4
            let dE: Vec3[];
            if (pen) {
                const pg = penaltyGradient(vertices, edges, pen);
                dE = dETpe.map((v, i) => [v[0] + pg[i][0], v[1] + pg[i][1], v[2] + pg[i][2]]);
            } else {
                dE = dETpe;
            }
            // The objective for the echo paths below (converged / singular):
            // E_tpe plus active penalties — same composition as the line
            // search's Armijo energy (plan §2.4).
            const echoEnergy = (): number => {
                const e = timed('energy', () =>
                    calculateEnergy(vertices, edges, disjointPairs, alpha, beta, epsilon),
                );
                return pen ? e + penaltyEnergy(vertices, edges, pen) : e;
            };

            let gTilde: Vec3[];
            let residual: number;
            let frozen: FrozenSaddleOperator | undefined;
            try {
                // The frozen-capable solve IS the default solve (same numbers,
                // one factorization — gradient.ts); the returned operator is
                // simply ignored unless projectionMode === 'frozen'.
                // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 6)
                const solved = solveConstrainedGradientSetFrozen(
                    vertices,
                    edges,
                    disjointPairs,
                    alpha,
                    beta,
                    epsilon,
                    dE,
                    set,
                    opts.factorMode,
                );
                gTilde = solved.gTilde;
                residual = solved.residual;
                frozen = opts.projectionMode === 'frozen' ? solved.frozen : undefined;
                // §D14: the saddle solved — g̃ is the field the step used (converged,
                // rejected, or accepted all carry it). The singular catch below leaves
                // capturedField null. @see plan §D14 / issue #9.
                if (collectField) capturedField = gTilde;
            } catch {
                // Exactly singular saddle system (e.g. an isolated vertex → zero Ā rows).
                // Fold into the spec §C step 10 contract: reject, echo the input, report —
                // the frame loop auto-pauses on accepted:false instead of crashing.
                // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C (step 10)
                return {
                    vertices,
                    // E₀ reuse (Task 4): the echoed vertices ARE the input γ₀, so a
                    // provided energyBefore already equals the objective at γ₀ by its
                    // invariant — reuse it instead of recomputing.
                    // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 4)
                    energy: opts.energyBefore ?? echoEnergy(),
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
                    // E₀ reuse (Task 4): converged echoes γ₀ unchanged, so a provided
                    // energyBefore already equals the objective at γ₀ — reuse it.
                    // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 4)
                    energy: opts.energyBefore ?? echoEnergy(),
                    accepted: false,
                    converged: true,
                    stats: { tau: 0, residual, gradientL2Norm, projectionIterations: 0 },
                };
            }

            const result = timed('lineSearch', () =>
                lineSearchStepSet(
                    vertices,
                    edges,
                    disjointPairs,
                    alpha,
                    beta,
                    epsilon,
                    dE,
                    gTilde,
                    set,
                    // E₀ reuse (Task 4): forward the precomputed E₀ so the line
                    // search skips its own objective(γ₀); undefined → recompute.
                    // Frozen operator (Task 6) and penalties (5C): forwarded only
                    // when active — the spreads keep the default-path options
                    // object IDENTICAL to before (bit-identity guards in
                    // frozenProjection.test.ts and penalties.test.ts).
                    // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Tasks 4, 6)
                    // @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2.4
                    {
                        energyBefore: opts.energyBefore,
                        // An EXPLICIT factorMode must reach the reassemble
                        // projection solves too (a frozen operator already
                        // carries its kind); when unset, everything inherits
                        // solveSaddleFromA's default and the options object
                        // stays IDENTICAL to the pre-option build.
                        // @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (decision 4)
                        ...(frozen || opts.factorMode !== undefined
                            ? {
                                  projection: {
                                      ...(frozen ? { frozen } : {}),
                                      ...(opts.factorMode !== undefined
                                          ? { factorMode: opts.factorMode }
                                          : {}),
                                  },
                              }
                            : {}),
                        ...(pen ? { penalties: pen } : {}),
                    },
                ),
            );
            // On failure lineSearchStepSet already echoes the input vertices unchanged and
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
        },
    );
    // §D14: append the captured g̃ (or null on singular) ONLY when requested — same
    // opt-in append pattern as `timings`. Both absent ⇒ `return outcome` unchanged
    // (no extra allocation). @see plan §D14 / issue #9.
    if (collect) {
        const timings = timingsEnd();
        if (timings) {
            return collectField
                ? { ...outcome, timings, descentField: capturedField }
                : { ...outcome, timings };
        }
    }
    if (collectField) return { ...outcome, descentField: capturedField };
    return outcome;
}

/**
 * Back-compat x₀ signature: delegates to {@link sobolevStepSet} with the
 * barycenter-only set `[barycenterBlock(x0)]` — numerically bit-identical to
 * the pre-ConstraintSet implementation (regression-proven by the unmodified
 * stage-1 golden tests plus the back-compat bit-identity test, spec §3.2).
 *
 * `x0` is the FROZEN barycenter target (computed once per run start via
 * `barycenterTarget`, never from the current iterate — see the x0 contract on
 * `solveConstrainedGradient`).
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §3.2
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
    return sobolevStepSet(vertices, edges, disjointPairs, [barycenterBlock(x0)], opts);
}
