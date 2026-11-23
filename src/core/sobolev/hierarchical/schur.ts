/**
 * `prepareSchur` -- the `k = O(1)` constraint route of Phase 2H (spec §8.2): the FULL saddle
 * `[[Ā, Cᵀ], [C, 0]] · [x; λ] = [b; d]` solved with matrix-free preconditioned GMRES on the
 * `σ`-shifted SPD operator `A_σ` and a small dense Schur complement `S = C A_σ⁻¹ Cᵀ`.
 *
 * WHY THERE IS A `prepare` STEP. `W = A_σ⁻¹ Cᵀ` and the factored `S` depend only on `C`, not on the
 * right-hand side, and `C` is fixed for a step while the rhs is not: `gradient.ts` calls
 * `saddle.prepare(C)` once per step and the returned solver then serves the gradient rhs AND every
 * projection iterate of every τ-trial (`lineSearch.ts`'s frozen projection). So {@link prepareSchur}
 * pays `k` GMRES runs ONCE and each {@link SchurSolver.solve} costs ONE GMRES plus `O(k·3n)` flops.
 * Doing it the other way round multiplies the phase's cost by `k`.
 *
 * `k = 0` IS A FIRST-CLASS CASE (the empty `ConstraintSet`, constraints spec §9a): there is no `S`,
 * `solve` is one GMRES, and the result is `x = A_σ⁻¹ b`.
 *
 * `k = O(n)` IS REFUSED BY NAME ({@link SchurRouteError}). `edgeLengthsBlock` has `|E|` rows
 * (`constraintSet.ts:211-213`), and spec §8.1 measures that `|E|` fresh GMRES runs cost
 * `O(N²·iters)` -- worse than the dense path this phase replaces. That block belongs to the nullspace
 * route (spec §8.2, §11.2 row 9), which is NOT this file.
 *
 * THE `Ā` vs `A_σ` TRAP, SETTLED IN SOURCE AND BY THE ORACLE (test/sobolev/hierarchical/schur.test.ts).
 * The dense saddle is assembled on the UNSHIFTED `Ā` (`assembleAFlat`), which is SINGULAR: `Ā 1 = 0`
 * (nullity exactly 1 per coordinate block, MEASURED at n = 48 and 240; `A_bct 1 = 0` bit-exactly
 * too). GMRES can only invert `A_σ = Ā + (σ/n) 1 1ᵀ` per block. Writing `N = [1₀ 1₁ 1₂]` (`3n × 3`,
 * the constant vector of each coordinate block) and `Q = N Nᵀ`, `A_σ = Ā + (σ/n) Q`, and
 * `A_σ⁻¹ N = N/σ`. The naive Schur complement -- `S λ = C y − d`, `x = y − W λ` with `W = A_σ⁻¹ Cᵀ`
 * -- therefore solves the SHIFTED saddle `[[A_σ, Cᵀ], [C, 0]]`, which penalises `‖Nᵀ x‖` and so
 * disagrees with the dense path exactly when the true `x` has a non-zero per-block mean. MEASURED
 * (the `'naive-shift'` knock-out, n = 240): 1.5e-3 to 1.7e0 on EVERY pinning set (barycentre, pin,
 * barycentre+length, +pin), and it also shows in the returned `residual` (8.4e-2 against 4.6e-9 on a
 * pin). It is blind only where nothing pins the translations. (A barycentre block sitting exactly at
 * its own barycentre on the Z3-symmetric trefoil is a DEGENERATE cell where the naive formula
 * happens to agree to 7e-15 -- `λ ≈ 0` -- which is why the tests offset the targets.) The EXACT
 * reduction is
 *
 * ```
 *   Ā x + Cᵀ λ = b,  C x = d          (Ā N = 0, so solvability needs Nᵀ(b − Cᵀλ) = 0)
 *   x = (y − W λ) − N m'              y = A_σ⁻¹ b,  W = A_σ⁻¹ Cᵀ,  m' ∈ R³ the constants' amplitudes
 *   [ S   G ] [λ ]   [C y − d]        S = C W (k×k, SPD),  G = C N (k×3) = per-block row sums of C
 *   [ Gᵀ  0 ] [m'] = [  Nᵀ b ]
 * ```
 *
 * `Gᵀ λ = Nᵀ b` is the solvability condition (a constraint on `λ` alone, no solve needed) and
 * `G = C N` is analytic. It is solved by two nested Cholesky factorisations -- `S` (`k×k`) and
 * `T = Gᵀ S⁻¹ G` (`3×3`) -- both of which THROW on a non-positive pivot, so a rank-deficient `C`
 * (or one that pins the translations only partially) is reported, never inverted. The result is
 * independent of `σ` (every `σ` term cancels): `schur.test.ts` re-runs with `σ` scaled ×4 and ×0.25
 * and the solutions agree to ~1e-15 (the Krylov space of the non-constant modes does not see `σ`).
 *
 * AGREEMENT WITH THE DENSE PATH IS BOUNDED BY THE DENSE PATH, NOT BY THIS FILE. At n = 960 the
 * dense LDLᵀ `x` is itself 1.4e-10..6.4e-10 from the true solution (double-double iterative
 * refinement, which converges in one step), while this solver is 3e-12..9e-12 from it. So
 * "agrees with the dense path to 1e-10" cannot be met there by ANY solver, and tightening `tol` to
 * chase it buys nothing (the literal agreement is identical to 3 digits for `tol` 1e-11..1e-14).
 * The bar is asserted against the refined truth; the literal figure is logged.
 *
 * TRANSLATION GAUGE. When `G ≈ 0` (`k = 0`, or only translation-invariant rows such as
 * `totalLengthBlock`) the dense saddle is itself singular and its `x` is defined only up to a
 * per-block constant -- MEASURED: dense LDLᵀ and LU return DIFFERENT means (6.0e-2 vs 8.0e-2) for the
 * same `k = 0` system. There the solver returns the `A_σ` gauge (`Nᵀ x = Nᵀ b / σ`, ≈ 0 for an
 * energy gradient), `T` is not formed, and the exact reduction above degenerates to `S λ = C y − d`.
 * A PARTIAL pin (`0 < rank G < 3`) cannot arise from the catalogue (`barycenterBlock` and
 * `pointBlock` pin all three translations, the others none) and is refused.
 *
 * NON-CONVERGENCE IS NOT A SINGULAR MATRIX (spec A33, `optimizer.ts:448`/`lineSearch.ts:329`: only
 * {@link SolveNotCertifiedError} escapes the seam's `'singular_system'` catch). Every GMRES run is
 * checked against its TRUE backward error `η = ‖b − A_σx‖ / (‖A_σ‖‖x‖ + ‖b‖)` (never the recurrence
 * residual, which under-reports by 10⁸-10¹², `gmres.ts` module TSDoc) and a failure throws
 * {@link SchurConvergenceError}, which IS a `SolveNotCertifiedError`. A rank-deficient `C` is a plain
 * {@link SchurRankError}: that one really is a singular system.
 *
 * The base was `ExternalSolveError` when this module shipped, because that was then the only class
 * the seam let through; spec A35 item 6 ruled the neutral base in and row 8b introduced it. A CPU
 * GMRES that missed its certificate is NOT an external solve, so that base contradicted
 * `ExternalSolveError`'s own TSDoc and destroyed `instanceof ExternalSolveError` as the discriminant
 * the SPECIFIED GPU recovery keys on ("re-run on the CPU path", [DESIGN §4.5] trigger 3 -- specified,
 * and as of 2026-10-11 not yet implemented: `src/gpu/driver.ts` has no `catch`). Do not re-narrow.
 *
 * `src/core` must not import `src/gpu` (`gmres.ts` module TSDoc); `src/gpu/saddle.ts`'s
 * `schurWoodburySolve` is factor-based and was read for ideas only.
 *
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.1, §8.2, §8.3, §5.1, A33
 * @see https://arxiv.org/abs/2006.07859 §6.3 (no matrix to factor once the apply is hierarchical)
 * @issue utof/repulsive-test2#62
 */
import { operatorSaddleResidual, SolveNotCertifiedError } from '../linsolve';
import {
    applyAsigma,
    backwardError,
    estimateNorm2,
    type GmresResult,
    type GmresStop,
    gmres,
} from './gmres';
import type { HierarchicalOperator } from './operator';
import type { Preconditioner } from './precond';

/**
 * Largest `k` {@link prepareSchur} accepts by default: barycentre (3) + totalLength (1) + 4 pins (12).
 * Why a cap and not "no limit": `prepare` costs `k` GMRES runs, so the route is only the right one
 * while `k` is a small constant; `edgeLengthsBlock` has `|E| = n` rows (spec §8.1) and must go to
 * the nullspace route instead. Override with {@link SchurOptions.maxK}.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.1, §8.2
 * @issue utof/repulsive-test2#62
 */
export const SCHUR_MAX_K = 16;

/**
 * Default GMRES `tol` on the (left-preconditioned) RECURRENCE residual, per column. Why 1e-12:
 * MEASURED knee of the error against the refined truth (n = 960, DEFAULT_THETA, `schur.test.ts`'s
 * `tol` sweep): 1e-10 gives 2.2e-10..5.7e-10 (over the 1e-10 bar), 1e-11 gives 2.1e-11..7.3e-11,
 * 1e-12 gives 3.7e-12..7.6e-12, and 1e-13 / 1e-14 give the same 2.2e-12..3.2e-12 floor (κ·u).
 * 1e-12 is 13x under the bar, within 4x of the floor, and 46 iterations against 51 for 1e-13.
 * The recurrence is NOT a convergence criterion (spec §5.3) -- this is only the stopping rule;
 * {@link DEFAULT_ETA_ACCEPT} is what certifies a run.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.3, §8.3
 */
export const DEFAULT_SCHUR_TOL = 1e-12;

/**
 * Largest TRUE backward error `η` of any GMRES run that {@link prepareSchur} / {@link SchurSolver.solve}
 * will accept; above it they throw {@link SchurConvergenceError}. Why a backward error and not the
 * recurrence residual: spec §5.3 / `gmres.ts` module TSDoc. Why 1e-10: it is a CERTIFICATE of
 * convergence, not the accuracy target (that is {@link DEFAULT_SCHUR_TOL}). At the default `tol` the
 * measured `η` is 6e-13..2.3e-12 (n = 240..960), i.e. 40x of headroom against a false alarm, while
 * the error is 2-4x `η` above the floor (`schur.test.ts` `tol` sweep), so an accepted run is within
 * ~4e-10 of the truth even at the threshold.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.3, §7.3, §8.3
 */
export const DEFAULT_ETA_ACCEPT = 1e-10;

/**
 * Relative pivot floor of the two Cholesky factorisations: a pivot `p_j <= RANK_PIVOT_REL · S_jj`
 * means row `j` is numerically a combination of the rows before it. Why 1e-10: a duplicated
 * constraint row gives EXACTLY equal columns of `W` (same rhs, deterministic GMRES), hence a pivot
 * at roundoff (~1e-16 relative), while a legitimately conditioned catalogue set sits many decades
 * above -- both ends are asserted in `schur.test.ts`.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.3
 */
export const RANK_PIVOT_REL = 1e-10;

/**
 * `|Σ_i C[r][c·n+i]| / Σ|C[r]|` below this for EVERY `(r, c)` means `C` is translation-invariant
 * (`G = C N ≈ 0`): only `totalLengthBlock`'s rows qualify, whose `±T_I` terms cancel per edge.
 * MEASURED (`schur.test.ts`): 1e-17 for `totalLengthBlock`, at least 6.1e-1 over the barycentre, pin
 * and barycentre+pin sets -- the cut sits in the middle of a >16-decade gap.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.2
 */
export const TRANSLATION_INVARIANT_REL = 1e-10;

/**
 * `k` is outside the `O(1)` regime this route serves -- the `edgeLengthsBlock` case (`|E|` rows).
 * Extends {@link SolveNotCertifiedError} ON PURPOSE: a plain `Error` thrown from `prepare(C)` is
 * folded into `'singular_system'` by `optimizer.ts:448`, which would present a DISPATCH bug (a
 * `k = O(n)` set routed here) as a singular matrix. NOT `ExternalSolveError` (spec A35 item 6):
 * nothing external ran, so the GPU arm's specified "re-run on the CPU" recovery is the wrong one.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.1, §8.2
 * @issue utof/repulsive-test2#62
 */
export class SchurRouteError extends SolveNotCertifiedError {
    constructor(
        readonly k: number,
        readonly maxK: number,
        readonly n: number,
    ) {
        super(
            `prepareSchur: k=${k} constraint rows exceed maxK=${maxK} (n=${n}). This is the k = O(n) regime of ` +
                `edgeLengthsBlock (|E| rows): an iterative solve costs O(N²·iters) there, worse than the dense path ` +
                `(spec §8.1). Route it through the nullspace route (spec §8.2, §11.2 row 9), or raise maxK ` +
                `only if k really is O(1).`,
        );
        this.name = 'SchurRouteError';
    }
}

/**
 * A GMRES run inside the Schur solve did not reach {@link SchurOptions.etaAccept} (or produced a
 * non-finite iterate). Extends {@link SolveNotCertifiedError} ON PURPOSE -- Why: `optimizer.ts:448`
 * and `lineSearch.ts:329` let exactly that class escape their `'singular_system'` / `ok:false`
 * folds; a plain `Error` would report "GMRES did not converge" as "the matrix is singular"
 * (spec A33). NOT `ExternalSolveError` (spec A35 item 6): this GMRES ran in `src/core`, so the
 * external arm's specified "re-run on the CPU path" recovery would re-run the identical failing
 * solve.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.3, §8.3, A33
 * @issue utof/repulsive-test2#62
 */
export class SchurConvergenceError extends SolveNotCertifiedError {
    constructor(
        readonly phase: 'prepare' | 'solve',
        /** Constraint row whose `A_σ⁻¹ Cᵀ` column failed (`prepare`), or `-1` for the rhs solve. */
        readonly column: number,
        readonly record: SchurGmresRecord,
        readonly etaAccept: number,
    ) {
        super(
            `prepareSchur: GMRES (${phase}${column >= 0 ? `, column ${column}` : ''}) reached true backward error ` +
                `η=${record.eta} > etaAccept=${etaAccept} after ${record.iters} iterations (stop=${record.stop}, ` +
                `recurrence=${record.recurrence}). NOT a singular system -- the iteration did not converge.`,
        );
        this.name = 'SchurConvergenceError';
    }
}

/**
 * `S = C A_σ⁻¹ Cᵀ` (or `T = Gᵀ S⁻¹ G`) is not numerically positive definite: `C` is rank-deficient,
 * or only PARTIALLY pins the three translations. A plain `Error` -- this one really is a singular
 * saddle system, and the seam's `'singular_system'` fold is the right report.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.3
 * @issue utof/repulsive-test2#62
 */
export class SchurRankError extends Error {
    constructor(
        readonly matrix: 'S' | 'T' | 'C',
        message: string,
    ) {
        super(`prepareSchur: ${message}`);
        this.name = 'SchurRankError';
    }
}

/**
 * Knock-out defects, TEST ONLY (need `knockOut: true`). Why they exist in `src/`: a gate that cannot
 * be tripped reads exactly like one that passed (CLAUDE.md), and the only honest way to show that
 * `schur.test.ts` would catch each defect is to be able to inject it into the SHIPPED code path.
 *  - `'naive-shift'`: ignore the constant-mode reduction (`S λ = C y − d`, `x = y − W λ`) -- the
 *    `Ā` vs `A_σ` trap.
 *  - `'lambda-sign'`: `x = y + W λ − N m'`, i.e. the wrong sign on the multiplier term.
 *  - `'no-symmetrise'`: factor `S` as computed (lower triangle), not as `(S + Sᵀ)/2`.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.3
 */
export type SchurDefect = 'naive-shift' | 'lambda-sign' | 'no-symmetrise';

/**
 * Options of {@link prepareSchur}.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.3, §8.2, §8.3
 */
export interface SchurOptions {
    /** GMRES recurrence `tol`. Default {@link DEFAULT_SCHUR_TOL}. */
    tol?: number;
    /** GMRES `maxit`. Default `gmres`'s own (`min(600, 2·3n)`). */
    maxit?: number;
    /** Default {@link SCHUR_MAX_K}. */
    maxK?: number;
    /** Largest accepted TRUE backward error per GMRES run. Default {@link DEFAULT_ETA_ACCEPT}. */
    etaAccept?: number;
    /** `‖A_σ‖₂`; default {@link estimateNorm2} (30 matvecs, once per `prepare`). */
    normA2?: number;
    /** Required to pass `defect` (mirrors `precond.ts`'s and `apply.ts`'s knock-out guards). */
    knockOut?: boolean;
    defect?: SchurDefect;
    /**
     * The "preconditioner" IS `A_σ⁻¹` (the direct route's Cholesky solve). Each solve is then
     * ONE application of it instead of a GMRES run — which with an exact inverse converges in
     * one iteration anyway but spends two applications and a matvec getting there. The TRUE
     * backward error is still measured with one matvec and still checked against `etaAccept`,
     * so a caller that sets this on an inexact preconditioner is refused, not believed.
     * @see src/core/sobolev/hierarchical/directSolve.ts
     */
    exactInverse?: boolean;
}

/**
 * What one GMRES run did, measured against the TRUE residual.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.3, §8.3
 */
export interface SchurGmresRecord {
    iters: number;
    matvecs: number;
    precondApplies: number;
    stop: GmresStop;
    /** The recurrence residual at the stopping iterate (NOT a convergence criterion). */
    recurrence: number;
    /** `η = ‖b − A_σx‖ / (‖A_σ‖‖x‖ + ‖b‖)`, formed from `b` with the iteration's own matvec; 0 for `b = 0`. */
    eta: number;
    /** `iters === maxit`. */
    censored: boolean;
}

/**
 * Conditioning of one small SPD matrix, from a Jacobi eigensolve.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.3
 */
export interface SmallConditioning {
    lambdaMin: number;
    lambdaMax: number;
    cond: number;
}

/**
 * Everything `prepare` measured. Why a report and not just a solver: spec §8.3 item 2 warns that
 * GMRES on a non-symmetric operator converges "while every residual looks fine"; `S`'s asymmetry
 * and conditioning are the numbers that expose it.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.3
 */
export interface SchurPrepareReport {
    n: number;
    k: number;
    tol: number;
    etaAccept: number;
    normA2: number;
    /** One record per constraint row (`W` column), in row order. */
    columns: SchurGmresRecord[];
    /** `max|S_rs − S_sr| / max|S|` BEFORE symmetrising: the GMRES error in `W`, as seen by `S`. */
    sDefect: number;
    /** `S = C A_σ⁻¹ Cᵀ` as FACTORED (`k×k` row-major; symmetrised unless the knock-out says otherwise). */
    S: Float64Array;
    sConditioning: SmallConditioning | null;
    /** `true` when `C` pins all three translations (the exact reduction runs); `false` = gauge. */
    pinned: boolean;
    /**
     * `max_{r,c} |Σ_i C[r][c·n+i]| / Σ|C[r]|`, the quantity {@link TRANSLATION_INVARIANT_REL} cuts.
     * Reported so the threshold's margin is visible: 1e-17 for `totalLengthBlock`, at least 6.1e-1 for
     * the pinning sets (MEASURED, `schur.test.ts`).
     */
    translationSensitivity: number;
    tConditioning: SmallConditioning | null;
}

/**
 * Result of one {@link SchurSolver.solve}: the seam's `{ x, lambda, residual }` plus the GMRES record.
 * `residual` is `operatorSaddleResidual`'s `‖K z − rhs‖ / max(1, ‖rhs‖)` on the UNSHIFTED `Ā` -- the
 * same definition the dense path reports, formed from the real matvec and not from any recurrence.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.1, §8.3
 */
export interface SchurSolveResult {
    x: number[];
    lambda: number[];
    residual: number;
    /** The rhs solve `y = A_σ⁻¹ b`; `stop: 'zero-rhs'`, 0 iterations when `rhsTop = 0`. */
    gmres: SchurGmresRecord;
}

/**
 * The prepared solver: structurally a `HierarchicalSaddleSolver` (`linsolve.ts`), with the extras.
 * NOT RE-ENTRANT (the operator and preconditioner are not).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.1, §8.2
 */
export interface SchurSolver {
    solve(rhsTop: ArrayLike<number>, rhsBottom?: ArrayLike<number>): SchurSolveResult;
    report: SchurPrepareReport;
}

const norm2 = (x: ArrayLike<number>): number => {
    let s = 0;
    for (let i = 0; i < x.length; i++) s += x[i] * x[i];
    return Math.sqrt(s);
};

/**
 * Cholesky `M = L Lᵀ` of a `k×k` row-major SPD matrix reading the LOWER triangle, in a fresh array.
 * Throws {@link SchurRankError} when a pivot is `<= RANK_PIVOT_REL · M_jj` or non-finite (so NaN
 * cannot pass: every comparison with NaN is false and the negation is what throws).
 * Why a Cholesky and not a pseudo-inverse: spec §8.3 item 2 / the brief -- rank deficiency must be
 * DETECTED, not papered over.
 */
function cholesky(M: Float64Array, k: number, which: 'S' | 'T'): Float64Array {
    const L = new Float64Array(k * k);
    for (let j = 0; j < k; j++) {
        let p = M[j * k + j];
        for (let q = 0; q < j; q++) p -= L[j * k + q] * L[j * k + q];
        if (!(p > RANK_PIVOT_REL * M[j * k + j])) {
            throw new SchurRankError(
                which,
                `${which} (${k}×${k}) is not positive definite at pivot ${j}: p=${p}, diag=${M[j * k + j]} ` +
                    `(relative floor ${RANK_PIVOT_REL}). ${
                        which === 'S'
                            ? 'C is rank-deficient (a duplicated or dependent constraint row).'
                            : 'C pins the translations only partially (0 < rank(C N) < 3).'
                    }`,
            );
        }
        const d = Math.sqrt(p);
        L[j * k + j] = d;
        for (let i = j + 1; i < k; i++) {
            let s = M[i * k + j];
            for (let q = 0; q < j; q++) s -= L[i * k + q] * L[j * k + q];
            L[i * k + j] = s / d;
        }
    }
    return L;
}

/** Solves `L Lᵀ y = x` IN PLACE. */
function cholSolve(L: Float64Array, k: number, x: Float64Array): void {
    for (let i = 0; i < k; i++) {
        let s = x[i];
        for (let q = 0; q < i; q++) s -= L[i * k + q] * x[q];
        x[i] = s / L[i * k + i];
    }
    for (let i = k - 1; i >= 0; i--) {
        let s = x[i];
        for (let q = i + 1; q < k; q++) s -= L[q * k + i] * x[q];
        x[i] = s / L[i * k + i];
    }
}

/**
 * Eigenvalues of a symmetric `k×k` matrix by cyclic Jacobi rotations (`k <= SCHUR_MAX_K`, so
 * `O(k³)` per sweep is noise next to ONE GMRES). Only used to REPORT conditioning; it never decides
 * a rank (the Cholesky pivots do).
 */
function symmetricEigenvalues(M: Float64Array, k: number): number[] {
    const a = Float64Array.from(M);
    for (let sweep = 0; sweep < 60; sweep++) {
        let off = 0;
        let diag = 0;
        for (let i = 0; i < k; i++) {
            diag += a[i * k + i] * a[i * k + i];
            for (let j = i + 1; j < k; j++) off += a[i * k + j] * a[i * k + j];
        }
        if (off <= 1e-30 * (diag + off)) break;
        for (let p = 0; p < k; p++) {
            for (let q = p + 1; q < k; q++) {
                const apq = a[p * k + q];
                if (apq === 0) continue;
                const theta = (a[q * k + q] - a[p * k + p]) / (2 * apq);
                const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
                const c = 1 / Math.sqrt(t * t + 1);
                const s = t * c;
                for (let r = 0; r < k; r++) {
                    const arp = a[r * k + p];
                    const arq = a[r * k + q];
                    a[r * k + p] = c * arp - s * arq;
                    a[r * k + q] = s * arp + c * arq;
                }
                for (let r = 0; r < k; r++) {
                    const apr = a[p * k + r];
                    const aqr = a[q * k + r];
                    a[p * k + r] = c * apr - s * aqr;
                    a[q * k + r] = s * apr + c * aqr;
                }
            }
        }
    }
    const ev: number[] = [];
    for (let i = 0; i < k; i++) ev.push(a[i * k + i]);
    return ev.sort((x, y) => x - y);
}

function conditioningOf(M: Float64Array, k: number): SmallConditioning {
    const ev = symmetricEigenvalues(M, k);
    const lambdaMin = ev[0];
    const lambdaMax = ev[k - 1];
    return { lambdaMin, lambdaMax, cond: lambdaMax / lambdaMin };
}

/**
 * Builds the reusable solver for THIS step's constraint Jacobian `C` (`k × 3n`, coordinate-block
 * layout, set row order -- exactly `evaluateConstraintSet(...).C`). Costs `k` GMRES runs
 * (`W = A_σ⁻¹ Cᵀ`), the `k×k` Schur complement and its Cholesky factors; nothing else is deferred to
 * `solve` except ONE GMRES.
 *
 * `op` supplies `Ā` (unshifted) AND `σ`; `precond` MUST be the preconditioner built from the SAME
 * `op.sigma` (`makePreconditioner(…, op.sigma)`), or `A_σ` and `Ã⁻¹` disagree about the constant mode.
 * `alpha`/`beta`/`epsilon` live in `op` (it was prepared with them): a saddle built outside
 * `sobolevStepSet` must use the core's `DEFAULTS` (`optimizer.ts:23`), which `dispatch.ts` does not
 * forward.
 *
 * Throws {@link SchurRouteError} (`k > maxK`), {@link SchurRankError} (rank-deficient `C`),
 * {@link SchurConvergenceError} (a GMRES run did not reach `etaAccept`).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.2, §8.3, §5.1
 * @issue utof/repulsive-test2#62
 */
export function prepareSchur(
    op: HierarchicalOperator,
    precond: Pick<Preconditioner, 'apply' | 'n'> & Partial<Pick<Preconditioner, 'sigma'>>,
    C: number[][],
    opts: SchurOptions = {},
): SchurSolver {
    const n = op.n;
    const m = 3 * n;
    const k = C.length;
    const tol = opts.tol ?? DEFAULT_SCHUR_TOL;
    const etaAccept = opts.etaAccept ?? DEFAULT_ETA_ACCEPT;
    const maxK = opts.maxK ?? SCHUR_MAX_K;
    const defect = opts.defect;
    if (defect !== undefined && !opts.knockOut) {
        throw new Error(
            `prepareSchur: defect '${defect}' is a knock-out instrument; pass knockOut: true in a test`,
        );
    }
    if (k > maxK) throw new SchurRouteError(k, maxK, n);
    // `gmres` takes only `apply`/`n` (the preconditioner is injected), so a `Ã⁻¹` built with another
    // σ than the operator's would not be caught there; it would only show up as slow convergence.
    // @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §2.3 (σ is one number, shared)
    if (precond.sigma !== undefined && precond.sigma !== op.sigma) {
        throw new Error(
            `prepareSchur: the preconditioner was built with sigma=${precond.sigma}, the operator has sigma=${op.sigma}`,
        );
    }
    for (let r = 0; r < k; r++) {
        if (C[r].length !== m) {
            throw new RangeError(
                `prepareSchur: C[${r}] has length ${C[r].length}, expected 3n = ${m}`,
            );
        }
    }

    const normA2 = opts.normA2 ?? estimateNorm2((u, out) => applyAsigma(op, u, out), m).norm;
    const ax = new Float64Array(m);

    /** One checked GMRES run: `A_σ x = b`, certified by the TRUE backward error. */
    const solveShifted = (
        b: Float64Array,
        phase: 'prepare' | 'solve',
        column: number,
    ): { x: Float64Array; record: SchurGmresRecord } => {
        const bNorm = norm2(b);
        // The projection solve always has rhsTop = 0 (`lineSearch.ts:279`): the answer is x = 0 and
        // `gmres` would still spend one preconditioner apply before noticing, so skip it. The record
        // keeps `gmres`'s own `stop` name for the case.
        if (bNorm === 0) {
            return {
                x: new Float64Array(m),
                record: {
                    iters: 0,
                    matvecs: 0,
                    precondApplies: 0,
                    stop: 'zero-rhs',
                    recurrence: 0,
                    eta: 0,
                    censored: false,
                },
            };
        }
        let res: Pick<
            GmresResult,
            'x' | 'iters' | 'matvecs' | 'precondApplies' | 'stop' | 'recurrence' | 'censored'
        >;
        if (opts.exactInverse) {
            const x = new Float64Array(m);
            precond.apply(b, x);
            // Recorded as a zero-iteration run that met its tolerance: the
            // certificate below (η, from a true matvec) is what vouches for it.
            res = {
                x,
                iters: 0,
                matvecs: 0,
                precondApplies: 1,
                stop: 'tol',
                recurrence: new Float64Array(0),
                censored: false,
            };
        } else res = gmres(op, precond, b, { tol, maxit: opts.maxit });
        applyAsigma(op, res.x, ax);
        let rr = 0;
        for (let i = 0; i < m; i++) rr += (b[i] - ax[i]) ** 2;
        const eta = backwardError(Math.sqrt(rr), normA2, norm2(res.x), bNorm);
        const record: SchurGmresRecord = {
            iters: res.iters,
            matvecs: res.matvecs,
            precondApplies: res.precondApplies,
            stop: res.stop,
            recurrence: res.iters > 0 ? res.recurrence[res.iters - 1] : 0,
            eta,
            censored: res.censored,
        };
        // Negated `<=` so NaN fails: `eta > etaAccept` would let a NaN iterate through.
        if (!(eta <= etaAccept)) throw new SchurConvergenceError(phase, column, record, etaAccept);
        return { x: res.x, record };
    };

    // G = C N: the per-block row sums, and the L1 scale that makes "translation-invariant" scale-free.
    const G = new Float64Array(k * 3);
    let gRel = 0;
    for (let r = 0; r < k; r++) {
        let l1 = 0;
        for (let c = 0; c < 3; c++) {
            let s = 0;
            for (let i = 0; i < n; i++) {
                const v = C[r][c * n + i];
                s += v;
                l1 += Math.abs(v);
            }
            G[r * 3 + c] = s;
        }
        if (!(l1 > 0) || !Number.isFinite(l1)) {
            throw new SchurRankError('C', `C row ${r} is zero or non-finite (L1 norm ${l1})`);
        }
        for (let c = 0; c < 3; c++) gRel = Math.max(gRel, Math.abs(G[r * 3 + c]) / l1);
    }
    const pinned = k > 0 && gRel > TRANSLATION_INVARIANT_REL;

    // W = A_σ⁻¹ Cᵀ: k GMRES runs, the whole cost of prepare.
    const W: Float64Array[] = [];
    const columns: SchurGmresRecord[] = [];
    for (let r = 0; r < k; r++) {
        const { x, record } = solveShifted(Float64Array.from(C[r]), 'prepare', r);
        W.push(x);
        columns.push(record);
    }

    // S = C W. Exactly symmetric in exact arithmetic; its asymmetry is the GMRES error in W.
    const S = new Float64Array(k * k);
    for (let r = 0; r < k; r++) {
        for (let s = 0; s < k; s++) {
            let d = 0;
            for (let i = 0; i < m; i++) d += C[r][i] * W[s][i];
            S[r * k + s] = d;
        }
    }
    let sDefect = 0;
    let sMax = 0;
    for (let r = 0; r < k; r++) {
        for (let s = 0; s < k; s++) {
            sMax = Math.max(sMax, Math.abs(S[r * k + s]));
            sDefect = Math.max(sDefect, Math.abs(S[r * k + s] - S[s * k + r]));
        }
    }
    sDefect = sMax > 0 ? sDefect / sMax : 0;
    // Why symmetrise: Cholesky reads one triangle, so an unsymmetrised S silently factors a
    // DIFFERENT matrix than the one S λ = … is meant to invert (knock-out 'no-symmetrise').
    if (defect !== 'no-symmetrise') {
        for (let r = 0; r < k; r++) {
            for (let s = r + 1; s < k; s++) {
                const avg = 0.5 * (S[r * k + s] + S[s * k + r]);
                S[r * k + s] = avg;
                S[s * k + r] = avg;
            }
        }
    }
    const sConditioning = k > 0 ? conditioningOf(S, k) : null;
    const LS = k > 0 ? cholesky(S, k, 'S') : new Float64Array(0);

    // T = Gᵀ S⁻¹ G (3×3), the Schur complement of the constant-mode block.
    const useExact = pinned && defect !== 'naive-shift';
    const H = new Float64Array(k * 3); // S⁻¹ G, k×3 row-major
    let LT: Float64Array = new Float64Array(0);
    let tConditioning: SmallConditioning | null = null;
    if (useExact) {
        const col = new Float64Array(k);
        for (let c = 0; c < 3; c++) {
            for (let r = 0; r < k; r++) col[r] = G[r * 3 + c];
            cholSolve(LS, k, col);
            for (let r = 0; r < k; r++) H[r * 3 + c] = col[r];
        }
        const T = new Float64Array(9);
        for (let a = 0; a < 3; a++) {
            for (let c = 0; c < 3; c++) {
                let s = 0;
                for (let r = 0; r < k; r++) s += G[r * 3 + a] * H[r * 3 + c];
                T[a * 3 + c] = s;
            }
        }
        for (let a = 0; a < 3; a++) {
            for (let c = a + 1; c < 3; c++) {
                const avg = 0.5 * (T[a * 3 + c] + T[c * 3 + a]);
                T[a * 3 + c] = avg;
                T[c * 3 + a] = avg;
            }
        }
        tConditioning = conditioningOf(T, 3);
        LT = cholesky(T, 3, 'T');
    }

    const report: SchurPrepareReport = {
        n,
        k,
        tol,
        etaAccept,
        normA2,
        columns,
        sDefect,
        S,
        sConditioning,
        pinned,
        translationSensitivity: gRel,
        tConditioning,
    };

    const solve = (rhsTop: ArrayLike<number>, rhsBottom?: ArrayLike<number>): SchurSolveResult => {
        if (rhsTop.length !== m) {
            throw new RangeError(
                `SchurSolver.solve: rhsTop length ${rhsTop.length} does not match 3n = ${m}`,
            );
        }
        const d = rhsBottom ? Float64Array.from(rhsBottom) : new Float64Array(k);
        if (d.length !== k) {
            throw new RangeError(
                `SchurSolver.solve: rhsBottom length ${d.length} does not match C rows ${k}`,
            );
        }
        const b = Float64Array.from(rhsTop);
        const { x: y, record } = solveShifted(b, 'solve', -1);

        // u = S⁻¹ (C y − d)
        const u = new Float64Array(k);
        for (let r = 0; r < k; r++) {
            let s = 0;
            for (let i = 0; i < m; i++) s += C[r][i] * y[i];
            u[r] = s - d[r];
        }
        cholSolve(LS, k, u);

        const lambda = Float64Array.from(u);
        const mp = new Float64Array(3); // m' (the constants' amplitudes); stays 0 in the gauge
        if (useExact) {
            // Nᵀ b: per-block sums of the rhs.
            const nb = new Float64Array(3);
            for (let c = 0; c < 3; c++) {
                let s = 0;
                for (let i = 0; i < n; i++) s += b[c * n + i];
                nb[c] = s;
            }
            // m' = T⁻¹ (Gᵀ u − Nᵀ b);   λ = u − H m'
            for (let a = 0; a < 3; a++) {
                let s = -nb[a];
                for (let r = 0; r < k; r++) s += G[r * 3 + a] * u[r];
                mp[a] = s;
            }
            cholSolve(LT, 3, mp);
            for (let r = 0; r < k; r++) {
                let s = 0;
                for (let c = 0; c < 3; c++) s += H[r * 3 + c] * mp[c];
                lambda[r] = u[r] - s;
            }
        }

        // x = (y − W λ) − N m'   ('lambda-sign' flips ONLY the multiplier term)
        const sign = defect === 'lambda-sign' ? 1 : -1;
        const x = Float64Array.from(y);
        for (let r = 0; r < k; r++) {
            const lr = sign * lambda[r];
            if (lr !== 0) for (let i = 0; i < m; i++) x[i] += lr * W[r][i];
        }
        for (let c = 0; c < 3; c++) {
            const shift = -mp[c];
            if (shift !== 0) for (let i = 0; i < n; i++) x[c * n + i] += shift;
        }

        const xs = Array.from(x);
        const ls = Array.from(lambda);
        const residual = operatorSaddleResidual(
            op,
            C,
            [...xs, ...ls],
            [...Array.from(b), ...Array.from(d)],
        );
        return { x: xs, lambda: ls, residual, gmres: record };
    };

    return { solve, report };
}
