/**
 * `prepareNullspace` -- the `k = O(n)` constraint route of Phase 2H (spec §8.2): the `|E|`
 * `edgeLengths` rows are eliminated by an IMPLICIT nullspace projection, and whatever `O(1)` rows
 * sit beside them (`barycenterBlock`, `pointBlock`) are handed to {@link prepareSchur} ON THE
 * PROJECTED OPERATOR. §8.2 routes per block; this is the composition that makes a MIXED set --
 * `{barycenter, edgeLengths, pins}`, the one set §8.2 names that needs both routes at once -- a
 * single solve rather than two.
 *
 * WHY A NULLSPACE ROUTE AT ALL. `edgeLengthsBlock` has one row per edge, so `k = |E| = n` on a
 * closed ring (`constraintSet.ts` `edgeLengthsBlock`). `prepareSchur` costs `k` GMRES runs, so at
 * `k = O(n)` it costs `O(N²·iters)` -- worse than the dense path this phase replaces (spec §8.1).
 * Here the `O(n)` rows cost ZERO extra GMRES runs: they are absorbed into an `O(n)` projection
 * applied per iteration. `prepare` runs `kOther` GMRES runs (0 for the pure `edgeLengths` set) and
 * each `solve` runs exactly ONE.
 *
 * §8.3 ITEM 1 -- THE NULLSPACE IS NEVER FORMED. `V = null(C_edge)` is `2n`-dimensional, so a
 * `3n × 2n` basis is the thing this phase exists to avoid. `P = I − C_edgeᵀ (C_edge C_edgeᵀ)⁻¹ C_edge`
 * is applied instead, and every factor is `O(n)`:
 *  - `C_edge` and `C_edgeᵀ` are 6-nonzeros-per-row sparse applies built from the unit tangents, never
 *    from the `|E| × 3n` dense block (which is `3n²` f64 = **21.6 GB at N = 30 000** -- see DOUBTS).
 *  - `(C_edge C_edgeᵀ)⁻¹` is ONE {@link solveLaplacian} call. `C_edge C_edgeᵀ` is EXACTLY cyclic
 *    tridiagonal: row `r` is `+T_r` at `i1`, `−T_r` at `i2`, so rows `r` and `s` share a vertex only
 *    for `|r − s| = 1 (mod m)` and the off-band entries are identically `0.0` (MEASURED: zero
 *    entries with `|M_rs| > 0` at cyclic distance ≥ 2, at n = 48…480). Bands: `b[r] = 2|T_r|²`,
 *    `a[r] = −T_r·T_{r−1}`, `c[r] = −T_r·T_{r+1}`, both corners `−T_0·T_{m−1}`. So `laplacian.ts`'s
 *    GENERIC {@link TriSystem} describes it and no new solver is written; only
 *    `assembleCurveLaplacian` is Laplacian-specific.
 *
 * `C_edge N = 0` EXACTLY, AND THAT IS WHAT MAKES THE COMPOSITION LEGAL. With `N = [1₀ 1₁ 1₂]` the
 * per-coordinate-block constant basis, row `r`'s per-block sum is `T_r[c] − T_r[c] = 0` -- bit-exact,
 * not to roundoff (MEASURED `max|C_edge N| = 0.0`). Three consequences, all of which the `schur.ts`
 * reduction needs and all of which were verified in dense arithmetic at n ≤ 120 BEFORE this file
 * existed:
 *  1. `N ⊂ V`, so `P N = N` and `Nᵀ P = Nᵀ`.
 *  2. The nullity of `P Ā P` RESTRICTED TO `V` is still exactly 3, with the same basis `N`
 *     (MEASURED: `nullDim(P Ā P) = n + 3` on `R^3n` -- `n` zeros from `V^⊥` plus exactly 3 inside
 *     `V`). The enlargement of the nullspace is exactly `V^⊥ = range(I − P)`, which is orthogonal to
 *     `V` and is never entered by an iteration started in `V`.
 *  3. `P A_σ P N = σ N` (MEASURED 3.55e-14 absolute, 1.7e-16 relative), i.e. **`A_σ,V⁻¹ N = N/σ`
 *     holds on `V`** -- the single identity `schur.ts`'s exact reduction is built on. `G = Ĉ_o N`
 *     with `Ĉ_o = C_o P` equals `C_o N`, the SAME analytic `G`, and `Nᵀ b̂ = Nᵀ b` exactly because
 *     `Nᵀ Ā = 0`. So `Gᵀλ = Nᵀb` is the same solvability condition it is in `schur.ts`.
 * Dropping only the constant-mode correction (forward `defect: 'naive-shift'`) moves `x` by
 * **5.6e-2 … 3.8e+0** against 8e-15 clean, so this is not passing by luck.
 *
 * §8.3 ITEM 2 -- THE ORDER, AND THE THREE QUANTITIES THAT DECIDE IT. The spec warns that the order
 * of projection and preconditioning "decides whether the projected operator stays symmetric positive
 * definite" and that getting it wrong makes "GMRES converge on a non-symmetric operator while every
 * residual looks fine". Five orders were enumerated and measured densely at n = 48/96/120
 * (`scratch-9/e4-spd.ts`). The SHIPPED order is `P Ã⁻¹ P · P A_σ P`, and:
 *  - The operator being inverted, `Zᵀ(P A_σ P)Z` on an orthonormal basis `Z` of `V`, is SYMMETRIC
 *    (defect ≤ 1.5e-15) and POSITIVE DEFINITE (λ_min = 6.5e-1 / 3.3e-1 / 2.7e-1). The value that
 *    would trip that is λ_min ≤ 0, and it EXISTS one line away: without the σ-shift, `Zᵀ(P Ā P)Z`
 *    has exactly 3 zero eigenvalues.
 *  - The preconditioned product's spectrum ON `V` is real and positive -- λ_min = 1.0000 exactly
 *    (the constant modes `precond.ts` pins there), λ_max 59.8 / 69.5 / 72.2 -- shown through the
 *    `Lᵀ(ZᵀPÃ⁻¹PZ)L` similarity transform `denseSpectrum.ts`'s `cholesky` TSDoc documents.
 *  - ITS EUCLIDEAN SYMMETRIC DEFECT IS 1.4e-1 … 2.1e-1 AND MUST BE: a product of two SPD matrices is
 *    not symmetric. **Do not "fix" that number.** A gate asserting `symDefect(Ã⁻¹A) ≈ 0` on the
 *    preconditioned product rejects the correct order.
 * NOT projecting the preconditioner's output (`Ã⁻¹ P A_σ P`) is the disqualified order, and the ONLY
 * measurement that sees it is the leakage out of `V`: **4.1e-1 against 2.6e-13**, while its
 * restriction `Zᵀ(Ã⁻¹PAP)Z` is IDENTICAL to the correct order's to every digit (because `ZᵀP = Zᵀ`).
 * That is the §5.3 failure mode exactly -- the iterate leaves the constraint subspace while the
 * restricted operator looks perfect -- and it is reachable here as `defect: 'precond-unprojected'`.
 *
 * §8.3 ITEM 3 -- CONDITIONING. Projection does NOT add a penalty: `cond(P A_σ P|V) / cond(A_σ)` is
 * **0.960 ± 0.003** over n = 48…240 (Cauchy interlacing; measured so the ratio is known not to
 * drift), and the preconditioned `κ_eig` ratio saturates at 1.12. What the route DOES add is the
 * projection solve's own conditioning: `cond(C_edge C_edgeᵀ)` grows as **exactly `n²`** (4.771e+1 at
 * n = 48 → 1.234e+6 at n = 7 680, ratio 4.00 per doubling across 160× in n), reaching **≈1.88e+7 at
 * N = 30 000**, i.e. `κ·u_f32 ≈ 1.1`, which breaches `[G6]`'s 0.1 bound with a crossover at
 * **n ≈ 8 956**. The operator's own crossing is at **n ≈ 1 027** (`cond(A_σ)` grows `n^3.32`,
 * 1.389e+6 at n = 960), so the operator is the binding constraint by 8.8× in `n` and §6's
 * mixed-precision ruling is what both lean on. **The projection solve must stay f64 above
 * n ≈ 9 000**; it is cheap to keep there (2 Thomas sweeps against two hierarchical applies).
 *
 * TRANSLATION GAUGE, AND THE PURE `edgeLengths` SET IS ALWAYS IN IT. `C_edge N = 0` means the
 * `edgeLengths` rows pin no translation, so a set with no `barycenter`/`point` block is
 * TRANSLATION-SINGULAR: the dense saddle has `[N; 0]` in its kernel, the dense path returns a
 * factorisation-dependent translation (spec A35 item 3), and double-double iterative refinement of
 * the dense system DIVERGES (MEASURED `|δz|/|z|` = 1.8e-1 → 1.9e+0), so there is no refined truth to
 * compare against either. This route returns the same gauge `schur.ts` does, **`Nᵀ x = Nᵀ b / σ`**:
 * `Nᵀ x_p = (C_edge N)ᵀ w = 0` and `Nᵀ v = Nᵀ A_σ,V⁻¹ b̂ / 1 = Nᵀ b̂ / σ = Nᵀ b / σ`. Verified by its
 * σ-signature: scaling `σ` by ×4 / ×0.25 leaves `x` unchanged to 4e-14…1.2e-13 on every PINNED set
 * (translation included) and scales the per-block means by **exactly 1/σ** (0.2500 / 4.0000) on the
 * pure set, with σ-invariance restored to 8e-14 after demeaning. A rhs with `Nᵀ b ≠ 0` makes the
 * dense system INCONSISTENT rather than merely gauged; the gradient rhs is structurally mean-free
 * (the tangent-point energy is translation-invariant), measured means 1e-18.
 *
 * TOPOLOGY. The band construction needs edge `r` and edge `r+1 (mod m)` to share a vertex -- i.e.
 * the edge list must be ONE consistent cycle in edge order. That is checked here, in `O(m)`, because
 * the consequence of skipping it is a silently WRONG `C_edge C_edgeᵀ` rather than a throw.
 * `makeHierarchicalSaddle` applies a STRICTER natural-order check up front
 * (`HierarchicalSaddleUnsupportedError`) for the preconditioner's sake; this one is deliberately
 * scoped to this file's own matrix and is NOT a second topology convention. It throws
 * {@link NullspaceRouteError}, a {@link SolveNotCertifiedError}, and NOT
 * `HierarchicalSaddleUnsupportedError`: `gradient.ts` calls `prepare(C)` INSIDE the optimizer's try,
 * where `optimizer.ts`'s catch folds a plain `Error` into `'singular_system'` -- so a plain refusal
 * raised from `prepare` would present a routing or topology mistake as a singular matrix, which is
 * the mis-report `SchurRouteError` already exists to prevent (spec A33, A35 item 6).
 *
 * NOT RE-ENTRANT. The operator, the preconditioner and the laplacian workspace each own shared
 * scratch (`apply.ts` `HierarchicalA`, `precond.ts`, `laplacian.ts` `LaplacianWorkspace`), and this
 * solver adds its own. One instance, one caller at a time.
 *
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.1, §8.2, §8.3, §5.1, A35, A36
 * @see https://arxiv.org/abs/2006.07859 §6.3 (no matrix to factor once the apply is hierarchical)
 * @issue utof/repulsive-test2#62
 */
import type { Edge, Vec3 } from '../../testConfigs';
import { SolveNotCertifiedError } from '../linsolve';
import { applyAsigma, estimateNorm2 } from './gmres';
import {
    type LaplacianWorkspace,
    makeLaplacianWorkspace,
    solveLaplacian,
    type TriSystem,
    triResidual,
} from './laplacian';
import type { HierarchicalOperator } from './operator';
import type { Preconditioner } from './precond';
import {
    prepareSchur,
    SCHUR_MAX_K,
    type SchurDefect,
    type SchurGmresRecord,
    type SchurPrepareReport,
} from './schur';

/**
 * The `O(n)` block has no structure this route can use, or the `O(1)` rows are not `O(1)`.
 * Extends {@link SolveNotCertifiedError} ON PURPOSE -- see the module TSDoc's TOPOLOGY paragraph for
 * why a plain `Error` here would be reported as `'singular_system'`.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.1, §8.2
 * @issue utof/repulsive-test2#62
 */
export class NullspaceRouteError extends SolveNotCertifiedError {
    constructor(
        readonly reason: 'cycle' | 'shape' | 'maxK',
        message: string,
    ) {
        super(`prepareNullspace: ${message}`);
        this.name = 'NullspaceRouteError';
    }
}

/**
 * Knock-out defects, TEST ONLY (need `knockOut: true`). Why they live in `src/`: a gate that cannot
 * be tripped reads exactly like one that passed (CLAUDE.md), and the only honest way to show that
 * the tests would catch each defect is to be able to inject it into the SHIPPED path -- the pattern
 * `precond.ts`, `apply.ts` and `schur.ts` already use.
 *  - `'precond-unprojected'`: skip the projection of `Ã⁻¹`'s output, i.e. the DISQUALIFIED order
 *    `Ã⁻¹ P A_σ P` of §8.3 item 2. MEASURED leakage out of `V`: 4.1e-1.
 *  - `'skip-particular'`: set `x_p = 0`, i.e. forget that `C_edge x = d_edge` needs a particular
 *    solution. Invisible to any bar that only looks at `x`'s accuracy on a `d_edge = 0` cell.
 * `'naive-shift'` is NOT duplicated here -- it is forwarded to `prepareSchur` through
 * {@link NullspaceOptions.schurDefect}, because the constant-mode reduction is that file's code.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.3
 */
export type NullspaceDefect = 'precond-unprojected' | 'skip-particular';

/**
 * The `O(n)` `edgeLengths` block, described SPARSELY. Never the `|E| × 3n` dense rows: that array is
 * `3n²` f64 by itself, 22 MB at n = 960 and 21.6 GB at N = 30 000.
 *
 * `vertices` and `edges` MUST be the same geometry `op` was built from -- the tangents are
 * re-derived here with `edgeLengthsBlock`'s formula, so a mismatch produces a `C_edge` for a
 * different curve, which nothing downstream can detect (the same unenforced "same vertices" rule as
 * `linsolve.ts`'s `A`, issue utof/repulsive-test2#83).
 * @see src/core/sobolev/constraintSet.ts (`edgeLengthsBlock`)
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.1
 */
export interface NullspaceEdgeBlock {
    vertices: Vec3[];
    edges: Edge[];
    /**
     * Index of the FIRST `edgeLengths` row in the stacked constraint set, so `lambda` comes back in
     * the set's own row order. `evaluateConstraintSet` concatenates blocks in array order, so the
     * `|E|` rows are contiguous at `[rowStart, rowStart + |E|)`.
     */
    rowStart: number;
}

/**
 * Options of {@link prepareNullspace}. Every numeric default is INHERITED from `schur.ts`; none is
 * re-chosen here (spec §8.3's non-goals: `tol` is ruled `DEFAULT_SCHUR_TOL`).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.2, §8.3
 */
export interface NullspaceOptions {
    /** GMRES recurrence `tol`. Default `schur.ts`'s `DEFAULT_SCHUR_TOL`. */
    tol?: number;
    /** GMRES `maxit`. Default `gmres`'s own. */
    maxit?: number;
    /** Largest `O(1)` row count. Default {@link SCHUR_MAX_K}; the `|E|` rows do not count. */
    maxK?: number;
    /** Largest accepted TRUE backward error per GMRES run. Default `schur.ts`'s. */
    etaAccept?: number;
    /**
     * `‖P A_σ P‖₂`. Default {@link estimateNorm2} on the PROJECTED operator (30 matvecs, once).
     * A cached UNPROJECTED `‖A_σ‖₂` must not be substituted: it is the larger of the two
     * (`‖P A_σ P‖₂ ≤ ‖A_σ‖₂`), so it makes `η` smaller and the certificate LOOSER. The measured gap
     * is tiny (λ_max 51.333 vs 51.370 at n = 48) but the direction is the unsafe one.
     */
    normA2?: number;
    /** Required to pass `defect` / `schurDefect`. */
    knockOut?: boolean;
    defect?: NullspaceDefect;
    /** Forwarded to {@link prepareSchur} (`'naive-shift'` is the constant-mode knock-out). */
    schurDefect?: SchurDefect;
}

/**
 * The projector's own health, measured per `prepare` on a deterministic probe set (no `Math.random`
 * in a gate). Reported, not thresholded: these are `O(κ·u)` quantities and a fixed bar on them would
 * be either vacuous in f64 or wrong in f32. The tests assert them against measured values and trip
 * each one with a knock-out.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.3
 */
export interface ProjectorReport {
    /** `max ‖P(Px) − Px‖ / ‖x‖` -- idempotence. */
    idempotenceDefect: number;
    /** `max ‖C_edge P x‖ / ‖x‖` -- the rows `P` must annihilate. */
    annihilationDefect: number;
    /** `max |xᵀ(Py) − yᵀ(Px)| / (‖x‖‖y‖)` -- self-adjointness. */
    selfAdjointDefect: number;
    /** Worst relative `triResidual` of the probe's `solveLaplacian` calls. */
    triResidual: number;
    /** `cond`-free structural check: the smallest `|b[r]|` of the band (`2|T_r|²`, so ≈2). */
    minDiagonal: number;
}

/**
 * Everything `prepare` measured.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.2, §8.3
 */
export interface NullspacePrepareReport {
    n: number;
    /** `|E|`, the `O(n)` rows absorbed by the projection. They cost ZERO GMRES runs. */
    kEdge: number;
    /** The `O(1)` rows routed to {@link prepareSchur}. `prepare` costs this many GMRES runs. */
    kOther: number;
    edgeRowStart: number;
    /** `‖P A_σ P‖₂`, as used by the `η` certificate. */
    normA2: number;
    projector: ProjectorReport;
    /**
     * {@link prepareSchur}'s report for the `O(1)` rows ON THE PROJECTED OPERATOR. At `kOther = 0`
     * (the pure `edgeLengths` set) it is still present with `k: 0`, `columns: []`,
     * `sConditioning: null`, `pinned: false` -- there is no Schur complement, and `prepareSchur`'s
     * own `k = 0` path is a first-class case (`schur.ts` module TSDoc), so this route does not
     * special-case it.
     */
    schur: SchurPrepareReport;
    /**
     * Cumulative {@link project} calls (= `solveLaplacian` calls) since construction, counting the
     * `‖P A_σ P‖₂` estimate, the projector probe, the `O(1)` row projections and every GMRES apply.
     * A structural counter, not a diagnostic: this route's whole cost claim (§8.1) is "`O(n)` per
     * apply, and the `|E|` rows cost ZERO extra GMRES runs", and the shape of that claim is the
     * DELTA per solve -- 2 per operator apply (input + output) plus 2 per preconditioner apply, so
     * 4 per GMRES iteration, plus 3 per `solve` outside the iteration (`x_p`, `b̂`, `λ_edge`). A
     * fourth projection per apply, or a projection that silently became a dense solve, would read
     * exactly like a passing test without it. Same reason as `Preconditioner.solves`.
     */
    projections: number;
}

/**
 * Result of one {@link NullspaceSolver.solve}.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.1, §8.3
 */
export interface NullspaceSolveResult {
    x: number[];
    /** Full stacked order: the `O(1)` rows in their original positions, the `|E|` rows at `rowStart`. */
    lambda: number[];
    /**
     * `‖K z − rhs‖₂ / max(1, ‖rhs‖₂)` on the UNSHIFTED `Ā`, the same definition
     * `operatorSaddleResidual` reports -- but formed STRUCTURALLY in `O(n)` from the sparse `C_edge`,
     * because `operatorSaddleResidual` needs the `|E| × 3n` dense `C` this route exists not to build.
     * Mathematically identical, NOT bit-identical (different summation order); the tests assert
     * agreement with `operatorSaddleResidual` at small `n`.
     */
    residual: number;
    /** The one GMRES run on the projected operator. */
    gmres: SchurGmresRecord;
    /** `‖C_edge x − d_edge‖₂ / max(1, ‖d_edge‖₂)` -- the `O(n)` rows' own satisfaction. */
    edgeConstraintResidual: number;
}

/**
 * The prepared solver. Structurally a `HierarchicalSaddleSolver` (`linsolve.ts`) with extras, so it
 * is assignable to the seam type. NOT RE-ENTRANT.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.1, §8.2
 */
export interface NullspaceSolver {
    solve(rhsTop: ArrayLike<number>, rhsBottom?: ArrayLike<number>): NullspaceSolveResult;
    report: NullspacePrepareReport;
}

const norm2 = (x: ArrayLike<number>): number => {
    let s = 0;
    for (let i = 0; i < x.length; i++) s += x[i] * x[i];
    return Math.sqrt(s);
};

/**
 * Unit tangents, `3m`, in the SAME spelling `edgeLengthsBlock` uses: `T_I = (γ_{i2} − γ_{i1}) / ℓ_I`
 * with the raw geometric `ℓ_I` (NO `+ε` -- constraints are geometric, `constraintSet.ts`'s
 * `edgeLengths` anchor) and the `ℓ < 1e-14` degenerate guard that zeroes the row.
 *
 * It is a SECOND spelling of one formula, which is a real risk; the test asserts that the sparse
 * `C_edge` built from this applies identically to `edgeLengthsBlock`'s dense rows.
 * @see src/core/sobolev/constraintSet.ts (`edgeLengthsBlock`)
 * @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §2
 */
function edgeTangents(vertices: Vec3[], edges: Edge[]): Float64Array {
    const T = new Float64Array(3 * edges.length);
    for (let r = 0; r < edges.length; r++) {
        const [i1, i2] = edges[r];
        const ex = vertices[i2][0] - vertices[i1][0];
        const ey = vertices[i2][1] - vertices[i1][1];
        const ez = vertices[i2][2] - vertices[i1][2];
        const ell = Math.sqrt(ex * ex + ey * ey + ez * ez);
        // Degenerate guard, same constant as edgeLengthsBlock / barycenterPhiAndC: a zero row makes
        // `C_edge C_edgeᵀ` singular, which `solveLaplacian`'s zero-pivot throw then reports.
        if (ell < 1e-14) continue;
        const inv = 1 / ell;
        T[3 * r] = ex * inv;
        T[3 * r + 1] = ey * inv;
        T[3 * r + 2] = ez * inv;
    }
    return T;
}

/**
 * `C_edge C_edgeᵀ` as a cyclic {@link TriSystem}, in `O(m)`. See the module TSDoc for why the
 * off-band entries are identically zero and why `laplacian.ts`'s generic solver therefore applies.
 * NOTE the diagonal is `2|T_r|²`, which is 2 only to within 4 ulp -- a gate written `=== 2` fails.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.2
 */
function edgeGramTri(T: Float64Array, m: number): TriSystem {
    const dot = (r: number, s: number) =>
        T[3 * r] * T[3 * s] + T[3 * r + 1] * T[3 * s + 1] + T[3 * r + 2] * T[3 * s + 2];
    const a = new Float64Array(m);
    const b = new Float64Array(m);
    const c = new Float64Array(m);
    for (let r = 0; r < m; r++) b[r] = 2 * dot(r, r);
    for (let r = 1; r < m; r++) a[r] = -dot(r, r - 1);
    for (let r = 0; r < m - 1; r++) c[r] = -dot(r, r + 1);
    const corner = -dot(0, m - 1);
    return { n: m, a, b, c, cornerTR: corner, cornerBL: corner, closed: true };
}

/**
 * The band construction's own precondition: the edge list is ONE consistent cycle IN EDGE ORDER, so
 * that edges `r` and `r ± 1 (mod m)` are exactly the pairs sharing a vertex. See the module TSDoc's
 * TOPOLOGY paragraph for why this is checked here and why it is not a second topology convention.
 */
function assertEdgeCycle(vertices: Vec3[], edges: Edge[]): void {
    const n = vertices.length;
    const m = edges.length;
    if (m < 3) {
        throw new NullspaceRouteError(
            'cycle',
            `need >= 3 edges for a cyclic tridiagonal C_edge C_edgeᵀ, got ${m}; route this curve to the DENSE path`,
        );
    }
    const degree = new Uint8Array(n);
    for (let r = 0; r < m; r++) {
        const [i1, i2] = edges[r];
        if (!(i1 >= 0 && i1 < n && i2 >= 0 && i2 < n) || i1 === i2) {
            throw new NullspaceRouteError(
                'cycle',
                `edge ${r} = [${i1},${i2}] is out of range or a self-loop (|V| = ${n})`,
            );
        }
        const next = edges[(r + 1) % m];
        if (i2 !== next[0]) {
            throw new NullspaceRouteError(
                'cycle',
                `edge ${r} = [${i1},${i2}] does not hand off to edge ${(r + 1) % m} = [${next[0]},${next[1]}]: ` +
                    `the edge list must be ONE consistent cycle in edge order, or C_edge C_edgeᵀ is not ` +
                    `cyclic tridiagonal and this route would silently project onto the wrong subspace. ` +
                    `Route this curve to the DENSE path (spec §8.2).`,
            );
        }
        degree[i1]++;
        degree[i2]++;
    }
    for (let i = 0; i < n; i++) {
        if (degree[i] !== 2) {
            throw new NullspaceRouteError(
                'cycle',
                `vertex ${i} has degree ${degree[i]}, not 2 — the edge list is not a single simple cycle ` +
                    `covering every vertex. Route this curve to the DENSE path (spec §8.2).`,
            );
        }
    }
}

/**
 * Builds the reusable solver for THIS step's constraint set, split into its `O(n)` `edgeLengths`
 * block and its `O(1)` rows. Costs `otherRows.length` GMRES runs (ZERO for the pure `edgeLengths`
 * set) plus `O(n)`; every `solve` costs ONE GMRES plus two unshifted `Ā` applies.
 *
 * `op` supplies `Ā` (unshifted) AND `σ`; `precond` MUST be built from the SAME `op.sigma`.
 * `otherRows` are the stacked set's non-`edgeLengths` rows IN THEIR ORIGINAL RELATIVE ORDER, each
 * `3n` long -- exactly `evaluateConstraintSet(...).C` with the `|E|` edge rows removed.
 *
 * Throws {@link NullspaceRouteError} (bad cycle, bad shape, `kOther > maxK`), and whatever
 * {@link prepareSchur} throws for the `O(1)` rows (`SchurRankError`, `SchurConvergenceError`).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.2, §8.3, §5.1
 * @issue utof/repulsive-test2#62
 */
export function prepareNullspace(
    op: HierarchicalOperator,
    precond: Pick<Preconditioner, 'apply' | 'n'> & Partial<Pick<Preconditioner, 'sigma'>>,
    edgeBlock: NullspaceEdgeBlock,
    otherRows: readonly number[][] = [],
    opts: NullspaceOptions = {},
): NullspaceSolver {
    const n = op.n;
    const dim = 3 * n;
    const { vertices, edges, rowStart } = edgeBlock;
    const defect = opts.defect;
    if ((defect !== undefined || opts.schurDefect !== undefined) && !opts.knockOut) {
        throw new Error(
            `prepareNullspace: defect '${defect ?? opts.schurDefect}' is a knock-out instrument; pass knockOut: true in a test`,
        );
    }
    if (vertices.length !== n) {
        throw new NullspaceRouteError(
            'shape',
            `edgeBlock.vertices has ${vertices.length} entries, the operator has n = ${n}. The tangents are ` +
                `re-derived from these vertices, so a mismatch builds C_edge for a different curve (issue #83).`,
        );
    }
    assertEdgeCycle(vertices, edges);
    const m = edges.length;
    const kOther = otherRows.length;
    const maxK = opts.maxK ?? SCHUR_MAX_K;
    if (kOther > maxK) {
        throw new NullspaceRouteError(
            'maxK',
            `${kOther} non-edgeLengths rows exceed maxK = ${maxK}; those rows go to the Schur complement, ` +
                `which costs one GMRES run each (spec §8.2)`,
        );
    }
    for (let r = 0; r < kOther; r++) {
        if (otherRows[r].length !== dim) {
            throw new NullspaceRouteError(
                'shape',
                `otherRows[${r}] has length ${otherRows[r].length}, expected 3n = ${dim}`,
            );
        }
    }
    const k = m + kOther;
    if (!(Number.isInteger(rowStart) && rowStart >= 0 && rowStart <= kOther)) {
        throw new NullspaceRouteError(
            'shape',
            `edgeBlock.rowStart = ${rowStart} must be an integer in [0, ${kOther}] (the |E| rows are ` +
                `contiguous, so at most ${kOther} O(1) rows can precede them)`,
        );
    }

    const T = edgeTangents(vertices, edges);
    const tri = edgeGramTri(T, m);
    const ws: LaplacianWorkspace = makeLaplacianWorkspace(m);
    const cBuf = new Float64Array(m);
    const cSol = new Float64Array(m);
    const projBuf = new Float64Array(dim);
    const applyBuf = new Float64Array(dim);
    let projections = 0;
    let worstTriResidual = 0;

    /** `out = C_edge u` (`m` out, `3n` in). 6 nonzeros per row. */
    const applyCe = (u: ArrayLike<number>, out: Float64Array): void => {
        for (let r = 0; r < m; r++) {
            const [i1, i2] = edges[r];
            let s = 0;
            for (let c = 0; c < 3; c++) s += T[3 * r + c] * (u[c * n + i1] - u[c * n + i2]);
            out[r] = s;
        }
    };

    /** `out -= C_edgeᵀ y` in place (`3n`), the subtraction {@link project} needs. */
    const subCeT = (y: ArrayLike<number>, out: Float64Array): void => {
        for (let r = 0; r < m; r++) {
            const [i1, i2] = edges[r];
            const yr = y[r];
            for (let c = 0; c < 3; c++) {
                const t = T[3 * r + c] * yr;
                out[c * n + i1] -= t;
                out[c * n + i2] += t;
            }
        }
    };

    /**
     * `out = P u = u − C_edgeᵀ (C_edge C_edgeᵀ)⁻¹ C_edge u`. ONE `solveLaplacian`, `O(n)` total.
     * `out` MAY alias `u` (the copy happens before the subtraction).
     */
    const project = (u: ArrayLike<number>, out: Float64Array): void => {
        applyCe(u, cBuf);
        solveLaplacian(tri, cBuf, cSol, ws);
        projections++;
        if ((u as unknown) !== (out as unknown)) for (let i = 0; i < dim; i++) out[i] = u[i];
        subCeT(cSol, out);
    };

    // The projected operator and preconditioner -- §8.3 item 2's SHIPPED order `P Ã⁻¹ P · P A_σ P`.
    //
    // THE SHIFT NEEDS NO PROJECTION, AND THAT IS NOT AN OVERSIGHT. `gmres` forms `A_σ` through
    // `applyAsigma`, which adds `σ·(Σu_c/n)` per block to the output of `applyA` -- i.e. the rank-3
    // term `(σ/n) N Nᵀ u`, computed from the UNPROJECTED `u`. Because `C_edge N = 0` exactly we have
    // `Nᵀ(I − P) = 0`, hence `Nᵀ u = Nᵀ P u`, so that term already equals `(σ/n) N Nᵀ P u`, and its
    // value lies in `span(N) ⊂ V`, so it needs no output projection either. The composite is
    // therefore EXACTLY `P A_σ P` (MEASURED `|P A_σ P N − σ N| = 3.55e-14`). Projecting the shift a
    // second time would be harmless but would also hide that this identity is load-bearing.
    //
    // BOTH projections of `applyA` are kept even though `P(I − P) = 0` makes the INPUT one
    // algebraically redundant (MEASURED: the two orders agree to 13 digits). In floating point
    // `P² = P` holds only to 1.1e-15, and the re-projection is what bounds the iterate's drift out of
    // `V`; it costs 2 Thomas sweeps against this apply's two hierarchical ones. A future agent
    // deleting it as dead code would be right in exact arithmetic and wrong here.
    const projectedOp: HierarchicalOperator = {
        n,
        sigma: op.sigma,
        applyA: (u, out) => {
            project(u, projBuf);
            op.applyA(projBuf, out);
            project(out, out);
        },
    };
    const projectedPrecond = {
        n: precond.n,
        sigma: precond.sigma,
        apply: (r: ArrayLike<number>, out: Float64Array) => {
            project(r, projBuf);
            precond.apply(projBuf, out);
            // 'precond-unprojected' is the DISQUALIFIED order `Ã⁻¹ P A_σ P` (§8.3 item 2): leakage
            // out of V jumps to 4.1e-1 while the restricted operator is bit-identical to the correct
            // one's, so only a C_edge x residual sees it.
            if (defect !== 'precond-unprojected') project(out, out);
        },
    };

    const normA2 =
        opts.normA2 ?? estimateNorm2((u, out) => applyAsigma(projectedOp, u, out), dim).norm;

    // Projector health, on a deterministic probe set (no Math.random in a gate).
    const projector: ProjectorReport = (() => {
        let idem = 0;
        let annih = 0;
        let selfAdj = 0;
        const probes: Float64Array[] = [];
        let seed = 20261011 >>> 0;
        for (let p = 0; p < 2; p++) {
            const v = new Float64Array(dim);
            for (let i = 0; i < dim; i++) {
                seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
                v[i] = seed / 4294967296 - 0.5;
            }
            probes.push(v);
        }
        const pv: Float64Array[] = [];
        for (const v of probes) {
            const o = new Float64Array(dim);
            project(v, o);
            worstTriResidual = Math.max(worstTriResidual, triResidual(tri, cSol, cBuf));
            pv.push(o);
            const o2 = new Float64Array(dim);
            project(o, o2);
            let d = 0;
            for (let i = 0; i < dim; i++) d += (o2[i] - o[i]) ** 2;
            idem = Math.max(idem, Math.sqrt(d) / norm2(v));
            applyCe(o, cBuf);
            annih = Math.max(annih, norm2(cBuf) / norm2(v));
        }
        let xy = 0;
        let yx = 0;
        for (let i = 0; i < dim; i++) {
            xy += probes[0][i] * pv[1][i];
            yx += probes[1][i] * pv[0][i];
        }
        selfAdj = Math.abs(xy - yx) / (norm2(probes[0]) * norm2(probes[1]));
        let minDiagonal = Number.POSITIVE_INFINITY;
        for (let r = 0; r < m; r++) minDiagonal = Math.min(minDiagonal, Math.abs(tri.b[r]));
        return {
            idempotenceDefect: idem,
            annihilationDefect: annih,
            selfAdjointDefect: selfAdj,
            triResidual: worstTriResidual,
            minDiagonal,
        };
    })();

    // `Ĉ_o = C_o P`: the O(1) rows as they act ON V. `Ĉ_o N = C_o P N = C_o N = G`, the SAME analytic
    // G, so `prepareSchur`'s solvability condition and its pinned/gauge dispatch are unchanged.
    const projectedOther: number[][] = otherRows.map((row) => {
        const out = new Float64Array(dim);
        project(row, out);
        return Array.from(out);
    });

    const schurSolver = prepareSchur(projectedOp, projectedPrecond, projectedOther, {
        tol: opts.tol,
        maxit: opts.maxit,
        maxK,
        etaAccept: opts.etaAccept,
        normA2,
        knockOut: opts.knockOut,
        defect: opts.schurDefect,
    });

    const report: NullspacePrepareReport = {
        n,
        kEdge: m,
        kOther,
        edgeRowStart: rowStart,
        normA2,
        projector,
        schur: schurSolver.report,
        // A live getter, not a snapshot: the counter keeps rising across every later `solve`, which
        // is what makes the per-solve DELTA (the structural claim) readable.
        get projections() {
            return projections;
        },
    };

    const solve = (
        rhsTop: ArrayLike<number>,
        rhsBottom?: ArrayLike<number>,
    ): NullspaceSolveResult => {
        if (rhsTop.length !== dim) {
            throw new RangeError(
                `NullspaceSolver.solve: rhsTop length ${rhsTop.length} does not match 3n = ${dim}`,
            );
        }
        const d = rhsBottom ? Float64Array.from(rhsBottom) : new Float64Array(k);
        if (d.length !== k) {
            throw new RangeError(
                `NullspaceSolver.solve: rhsBottom length ${d.length} does not match |E| + O(1) rows = ${k}`,
            );
        }
        const b = Float64Array.from(rhsTop);
        const dEdge = d.subarray(rowStart, rowStart + m);

        // x_p: the minimum-norm particular solution of `C_edge x = d_edge`, one solveLaplacian.
        // `Nᵀ x_p = (C_edge N)ᵀ w = 0`, which is what makes the gauge statement `Nᵀx = Nᵀb/σ` exact.
        const xp = new Float64Array(dim);
        if (defect !== 'skip-particular') {
            cBuf.set(dEdge);
            solveLaplacian(tri, cBuf, cSol, ws);
            projections++;
            // xp = +C_edgeᵀ w, and subCeT subtracts, so negate once.
            for (let r = 0; r < m; r++) cBuf[r] = -cSol[r];
            subCeT(cBuf, xp);
        }

        // b̂ = P (b − Ā x_p). The UNSHIFTED Ā: the reduced problem on V is `P Ā P v + Ĉ_oᵀλ = b̂`.
        // Using A_σ here would re-introduce the shift the constant-mode reduction then removes.
        op.applyA(xp, applyBuf);
        const bHat = new Float64Array(dim);
        for (let i = 0; i < dim; i++) bHat[i] = b[i] - applyBuf[i];
        project(bHat, bHat);

        // d̂_o = d_o − C_o x_p, in the O(1) rows' own order.
        const dHat = new Float64Array(kOther);
        for (let r = 0, q = 0; r < k; r++) {
            if (r >= rowStart && r < rowStart + m) continue;
            let s = 0;
            const row = otherRows[q];
            for (let i = 0; i < dim; i++) s += row[i] * xp[i];
            dHat[q] = d[r] - s;
            q++;
        }

        const inner = schurSolver.solve(bHat, dHat);
        const x = new Float64Array(dim);
        for (let i = 0; i < dim; i++) x[i] = xp[i] + inner.x[i];

        // λ_edge = (C_edge C_edgeᵀ)⁻¹ C_edge (b − Ā x − C_oᵀ λ_o): the multipliers the projection
        // never formed. One more unshifted apply; it is what makes `lambda` a full stacked vector.
        op.applyA(x, applyBuf);
        const top = new Float64Array(dim);
        for (let i = 0; i < dim; i++) top[i] = b[i] - applyBuf[i];
        for (let q = 0; q < kOther; q++) {
            const lq = inner.lambda[q];
            if (lq !== 0) {
                const row = otherRows[q];
                for (let i = 0; i < dim; i++) top[i] -= row[i] * lq;
            }
        }
        applyCe(top, cBuf);
        solveLaplacian(tri, cBuf, cSol, ws);
        projections++;

        const lambda = new Array<number>(k).fill(0);
        for (let r = 0, q = 0; r < k; r++) {
            if (r >= rowStart && r < rowStart + m) lambda[r] = cSol[r - rowStart];
            else lambda[r] = inner.lambda[q++];
        }

        // Structural O(n) saddle residual, same definition as `operatorSaddleResidual`.
        let resSq = 0;
        let rhsSq = 0;
        const row0 = new Float64Array(dim);
        for (let q = 0; q < kOther; q++) {
            const lq = lambda[q < rowStart ? q : q + m];
            const row = otherRows[q];
            for (let i = 0; i < dim; i++) row0[i] += row[i] * lq;
        }
        for (let r = 0; r < m; r++) cBuf[r] = -cSol[r];
        const ceTLambda = new Float64Array(dim);
        subCeT(cBuf, ceTLambda);
        for (let i = 0; i < dim; i++) {
            const dd = applyBuf[i] + ceTLambda[i] + row0[i] - b[i];
            resSq += dd * dd;
            rhsSq += b[i] * b[i];
        }
        applyCe(x, cBuf);
        let edgeNum = 0;
        let edgeDen = 0;
        for (let r = 0; r < m; r++) {
            const dd = cBuf[r] - dEdge[r];
            resSq += dd * dd;
            edgeNum += dd * dd;
            edgeDen += dEdge[r] * dEdge[r];
            rhsSq += dEdge[r] * dEdge[r];
        }
        for (let q = 0, r = 0; r < k; r++) {
            if (r >= rowStart && r < rowStart + m) continue;
            let s = 0;
            const row = otherRows[q];
            for (let i = 0; i < dim; i++) s += row[i] * x[i];
            const dd = s - d[r];
            resSq += dd * dd;
            rhsSq += d[r] * d[r];
            q++;
        }

        return {
            x: Array.from(x),
            lambda,
            residual: Math.sqrt(resSq) / Math.max(1, Math.sqrt(rhsSq)),
            gmres: inner.gmres,
            edgeConstraintResidual: Math.sqrt(edgeNum) / Math.max(1, Math.sqrt(edgeDen)),
        };
    };

    return { solve, report };
}
