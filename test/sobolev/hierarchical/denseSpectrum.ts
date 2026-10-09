/**
 * Dense symmetric linear algebra for the Phase-2H hierarchical TESTS. **TEST ONLY** and O(n³) —
 * `src/` never forms a dense matrix (spec §1.1: no factorization at all in this phase).
 *
 * WHY IT EXISTS. Gate H5's three properties are spectral (`dim null(L^{2−s})`, `κ_eig(Ã⁻¹A_σ)` and
 * its growth in `n`), and until this module existed they could only be QUOTED from the Python
 * corpus — `.superpowers/sdd/2026-09-11-phase2h-queue/scratch/e4-closer-0b.json`, measured on the
 * lab's operators, not on the ones `src/` ships. This is the TypeScript stand-in for the oracle's
 * `numpy.linalg.eigvalsh`: Householder tridiagonalisation (Numerical Recipes §11.2 `tred2`,
 * eigenvalues only — no back-transformation) then Sturm-sequence inertia counting and bisection on
 * the tridiagonal (NR §11.4). It reproduces `e4-closer-0b.json`'s `kappaPrecond` to 8 significant
 * digits (`precond.test.ts`'s validation case), which is what licenses using it on the production
 * operator.
 *
 * WHY NOT the matrix-free inverse iteration spec §7.8 names ("Property 1 is an integer at any n
 * (inverse iteration, matrix-free, O(N) storage)"): inverse iteration needs a SOLVE with
 * `L^{2−s}`, and this phase has no factorization — `L^{2−s}` is matvec-only. `lowOrder.test.ts`
 * carries the matrix-free alternative that does work on the production variant, and the measured
 * reason it cannot replace this instrument for the knock-out is in the task-6 report.
 *
 * NON-SYMMETRIC PRODUCTS ARE REACHABLE TOO, which is why {@link cholesky} lives here: `Ã⁻¹A_σ` is
 * a product of two SPD matrices and is not symmetric, but with `A_σ = R Rᵀ` it is SIMILAR to
 * `Rᵀ Ã⁻¹ R`, which is. See {@link cholesky} for the derivation and `precond.test.ts` for the use.
 *
 * @see https://numerical.recipes §11.2 (`tred2`), §11.4 (Sturm sequence / bisection)
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.8, §11.2 task 6
 * @issue utof/repulsive-test2#59
 */

/** `e4_precond_integrity.py:33`: `rank_tol = 1e-8 * max(|λ_max|, 1e-300)`. */
export const RANK_TOL_REL = 1e-8;

/** A symmetric tridiagonal, the form both the eigensolver and the matrix-free Lanczos produce. */
export interface Tridiag {
    /** Diagonal, length `n`. */
    d: Float64Array;
    /** Sub-diagonal, length `n − 1`. */
    e: Float64Array;
}

/**
 * NR §11.2 `tred2` with `yesvecs = false`. `A` (row-major, **lower triangle read**) is DESTROYED.
 *
 * Why lower-triangle-only matters: a matrix densified column-by-column from a matvec is symmetric
 * only to roundoff, and reading one triangle is the symmetrisation — an asymmetry of O(u) cannot
 * perturb the eigenvalues it ignores.
 * @see https://numerical.recipes §11.2
 */
function tridiagonalize(A: Float64Array, n: number): Tridiag {
    const d = new Float64Array(n);
    const e = new Float64Array(n);
    for (let i = n - 1; i >= 1; i--) {
        const l = i - 1;
        let h = 0;
        let scale = 0;
        if (l > 0) {
            for (let k = 0; k <= l; k++) scale += Math.abs(A[i * n + k]);
            if (scale === 0) {
                e[i] = A[i * n + l];
            } else {
                for (let k = 0; k <= l; k++) {
                    A[i * n + k] /= scale;
                    h += A[i * n + k] * A[i * n + k];
                }
                let f = A[i * n + l];
                let g = f >= 0 ? -Math.sqrt(h) : Math.sqrt(h);
                e[i] = scale * g;
                h -= f * g;
                A[i * n + l] = f - g;
                f = 0;
                for (let j = 0; j <= l; j++) {
                    g = 0;
                    for (let k = 0; k <= j; k++) g += A[j * n + k] * A[i * n + k];
                    for (let k = j + 1; k <= l; k++) g += A[k * n + j] * A[i * n + k];
                    e[j] = g / h;
                    f += e[j] * A[i * n + j];
                }
                const hh = f / (h + h);
                for (let j = 0; j <= l; j++) {
                    f = A[i * n + j];
                    g = e[j] - hh * f;
                    e[j] = g;
                    for (let k = 0; k <= j; k++) A[j * n + k] -= f * e[k] + g * A[i * n + k];
                }
            }
        } else {
            e[i] = A[i * n + l];
        }
        d[i] = h;
    }
    for (let i = 0; i < n; i++) d[i] = A[i * n + i];
    return { d, e: e.subarray(1, n) };
}

/** Number of eigenvalues of a symmetric tridiagonal strictly below `tau` (LDLᵀ inertia, NR §11.4). */
function countBelow(t: Tridiag, tau: number): number {
    const { d, e } = t;
    const n = d.length;
    let piv = d[0] - tau;
    let c = piv < 0 ? 1 : 0;
    for (let i = 1; i < n; i++) {
        if (piv === 0) {
            // Standard zero-pivot perturbation; a hard 0 would make the next step non-finite.
            piv = -2.220446049250313e-16 * (Math.abs(e[i - 1]) + Math.abs(d[i]) + 1e-300);
        }
        piv = d[i] - tau - (e[i - 1] * e[i - 1]) / piv;
        if (piv < 0) c++;
    }
    return c;
}

/**
 * `index`-th smallest eigenvalue (0-based) of a symmetric tridiagonal, by bisection on
 * {@link countBelow} over the Gershgorin interval.
 *
 * Why: the resolution floor is `2.3e-16 · span`, so a quantity of size `λ_max` is resolved to
 * ~1e-14 relative and a structural `λ = 1` next to `λ_max ≈ 70` is resolved to ~3e-14 ABSOLUTE —
 * which is the bar `precond.test.ts` asserts `λ₁ = 1` against, not `u_f64`.
 * @see https://numerical.recipes §11.4
 */
export function eigAt(t: Tridiag, index: number): number {
    const { d, e } = t;
    const n = d.length;
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < n; i++) {
        const r = (i > 0 ? Math.abs(e[i - 1]) : 0) + (i < n - 1 ? Math.abs(e[i]) : 0);
        lo = Math.min(lo, d[i] - r);
        hi = Math.max(hi, d[i] + r);
    }
    const span = Math.max(hi - lo, 1e-300);
    for (let it = 0; it < 300; it++) {
        const mid = 0.5 * (lo + hi);
        if (countBelow(t, mid) > index) hi = mid;
        else lo = mid;
        if (hi - lo <= 2.3e-16 * span) break;
    }
    return 0.5 * (lo + hi);
}

/** What {@link spectrum} reports — `e4_precond_integrity.py:50-53`'s row, in the same order. */
export interface Spectrum {
    lambda1: number;
    lambda2: number;
    lambdaMax: number;
    rankTol: number;
    nullDim: number;
}

/** `λ₁`, `λ₂`, `λ_max`, and `#{|λ| < 1e-8 λ_max}` — `e4_precond_integrity.py:32-34`'s recipe. */
export function spectrum(dense: Float64Array, n: number): Spectrum {
    const t = tridiagonalize(dense, n);
    const lambdaMax = eigAt(t, n - 1);
    const rankTol = RANK_TOL_REL * Math.max(Math.abs(lambdaMax), 1e-300);
    return {
        lambda1: eigAt(t, 0),
        lambda2: eigAt(t, 1),
        lambdaMax,
        rankTol,
        nullDim: countBelow(t, rankTol) - countBelow(t, -rankTol),
    };
}

/**
 * Lower-triangular Cholesky `A = L Lᵀ`, row-major, in a fresh array. Throws on a non-positive
 * pivot, which is the SPD check.
 *
 * Why this unlocks `κ_eig`: `Ã⁻¹A_σ` is a product of two SPD matrices, hence NOT symmetric, and
 * this repo has no non-symmetric eigensolver. But with `A_σ = L Lᵀ`,
 * `Lᵀ (Ã⁻¹ L Lᵀ) L⁻ᵀ = Lᵀ Ã⁻¹ L` — a similarity transform, so the same eigenvalues, and the
 * right-hand side IS symmetric because `Ã⁻¹` is. A square root is not needed, only a factor.
 * @see https://numerical.recipes §2.9
 */
export function cholesky(A: Float64Array, n: number): Float64Array {
    const L = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
            let s = A[i * n + j];
            for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
            if (i === j) {
                if (!(s > 0)) throw new Error(`cholesky: pivot ${s} at row ${i} — not SPD`);
                L[i * n + i] = Math.sqrt(s);
            } else {
                L[i * n + j] = s / L[j * n + j];
            }
        }
    }
    return L;
}

/** Solves `L Lᵀ x = x` IN PLACE, `L` lower-triangular from {@link cholesky}. */
export function cholSolve(L: Float64Array, n: number, x: Float64Array): void {
    for (let i = 0; i < n; i++) {
        let s = x[i];
        for (let k = 0; k < i; k++) s -= L[i * n + k] * x[k];
        x[i] = s / L[i * n + i];
    }
    for (let i = n - 1; i >= 0; i--) {
        let s = x[i];
        for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
        x[i] = s / L[i * n + i];
    }
}

/** `C = A B`, row-major, `k`-outer so a zero in `A` skips a whole row. Why: O(n³) at n=480. */
export function matmul(A: Float64Array, B: Float64Array, n: number): Float64Array {
    const C = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
        for (let k = 0; k < n; k++) {
            const a = A[i * n + k];
            if (a === 0) continue;
            for (let j = 0; j < n; j++) C[i * n + j] += a * B[k * n + j];
        }
    }
    return C;
}

/** `Aᵀ`, row-major, in a fresh array. Why: the similarity transform needs `Rᵀ M R`. */
export function transpose(A: Float64Array, n: number): Float64Array {
    const T = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) T[j * n + i] = A[i * n + j];
    }
    return T;
}

/**
 * Densifies a matvec column by column: `A[:,j] = mv(e_j)`. `n` applies, O(n) each for the
 * hierarchical operators, so this is the O(n²)-storage bridge from a matvec-only `src/` to a
 * spectral test. TEST ONLY — spec §2.3 forbids it in production (`n` applies, ~64 min at n=30 000).
 */
export function densify(n: number, mv: (u: Float64Array, out: Float64Array) => void): Float64Array {
    const A = new Float64Array(n * n);
    const e = new Float64Array(n);
    const col = new Float64Array(n);
    for (let j = 0; j < n; j++) {
        e.fill(0);
        e[j] = 1;
        mv(e, col);
        for (let i = 0; i < n; i++) A[i * n + j] = col[i];
    }
    return A;
}
