/**
 * The DIRECT variant of the matrix-free route: assemble the scalar |V|×|V| inner-product matrix
 * `A = B + B⁰` exactly, Cholesky-factor `A_σ = A + (σ/n)·11ᵀ` once, and hand the pair
 * (dense matvec, exact inverse) to the SAME constraint machinery the hierarchical route uses
 * (`prepareSchur` / `prepareNullspace`), where GMRES then converges in one iteration.
 *
 * Why it exists. The block cluster tree only pays when most edge pairs are far apart relative
 * to their clusters. On a tight tangle they are not: measured on the paper's Knot128 dataset
 * (`perturbed/knot0001`, 1 998 vertices, 2026-11-23) 96% of all pairs are near-field at the
 * start, so every one of the ~260 operator and ~260 preconditioner applications of a step is a
 * dense O(n²) sweep (0.20 s and 0.27 s each) and the step costs 83–120 s. The existing dense
 * route is no better there: it factors the full (3n+k)-dimensional saddle matrix, 142 s of a
 * 204 s step. But `Ā` is three copies of one n×n matrix, and the constraints are a handful of
 * rows, so ONE n×n Cholesky (n³/3 flops, seconds at n = 2 000) plus the Schur complement that
 * `schur.ts` already implements is all the linear algebra the step needs.
 *
 * What this is NOT: a path to n = 30 000. It is O(n²) memory (two n×n doubles) and O(n³) time,
 * so it is capped (`DIRECT_MAX_N`) and chosen per step only where it is cheaper than the tree
 * (`directIsCheaper`). The paper's remark that going hierarchical leaves "no matrix to factor"
 * (arXiv:2006.07859 §6.3) is about large n; this is the small-n / dense-tangle complement.
 *
 * The matrix is the one `innerProduct.ts` assembles (`assembleAFlat`), computed here in O(n²)
 * time with flat arrays; that file is the reference and golden surface and is not touched.
 * Agreement is to rounding, not bit-for-bit (the accumulation order differs) — tested.
 * @see src/core/sobolev/innerProduct.ts (assembleBHighFlat, assembleBLowFlat: the formulas)
 * @see src/core/sobolev/hierarchical/schur.ts (the Schur complement this feeds)
 */
import type { Edge, Vec3 } from '../../testConfigs';
import type { HierarchicalOperator } from './operator';

/**
 * Largest vertex count the direct variant accepts. Why 4 000: the matrix and its factor are
 * `2 · 8n²` bytes — 256 MB at 4 000 — and the factorization is n³/3 flops, 21e9 there.
 * The same number as the pair-list limit of the dense route, for the same O(n²) reason.
 * @see src/core/sobolevRoute.ts (PAIR_LIST_MAX_N)
 */
export const DIRECT_MAX_N = 4000;

/**
 * Is the direct route cheaper than the tree route for a ring of `n` vertices whose block lists
 * hold `nearPairs` near-field edge pairs?
 *
 * Both costs are a whole step's linear algebra (build, the constraint columns, the solves),
 * measured with `bench/scale/directVsTree.ts` on 2026-11-23
 * (`bench/results/2026-11-23-direct-vs-tree.jsonl`):
 *  - direct: 0.21 / 1.0 / 4.0 / 24 s at n = 500 / 1 000 / 2 000 / 4 000, fitted as
 *    `0.5 µs·n² + 0.25 ns·n³` (assembly and the dense solves; the factorization);
 *  - tree: 5.9–8.8 µs per near-field pair where the tree engages at all (25 µs on a pure
 *    tangle, where GMRES also needs five times the iterations); the rule uses 6.5 µs.
 * So direct wins when `nearPairs · 6.5 µs > 0.5 µs·n² + 0.25 ns·n³`, i.e. when the near-field
 * fraction `nearPairs / n²` exceeds `0.077 + 3.8e-5·n` — 9.6% at n = 500, 15.4% at 2 000,
 * 23% at 4 000. Checked against every row of that file: the rule picks the faster solver in
 * all of them, including the three a first, more eager fit (`0.017 + 2.2e-5·n`) got wrong
 * (near-field 6–9%, where the tree was still about a second faster).
 * Only the RATIO of the two costs enters, so a faster or slower machine moves both sides alike.
 * Not modelled: more constraint rows (pins) add solves to the tree route only, which favours
 * the direct route further.
 */
export function directIsCheaper(n: number, nearPairs: number): boolean {
    if (n > DIRECT_MAX_N) return false;
    return nearPairs > n * n * (0.077 + 3.8e-5 * n);
}

/**
 * The scalar inner-product matrix `A = B + B⁰` (row-major, symmetric, n×n), over every pair of
 * edges that share no vertex — the same pair set `calculateDisjointPairs` lists.
 *
 * How it gets from the reference's ~6 µs per pair to ~50 ns:
 *  - the fractional power is a function of a VERTEX pair, `(|γ_i − γ_j| + ε)^−(2σ+1)`, and each
 *    vertex pair is shared by up to four edge pairs, so it is tabulated once (`P`);
 *  - an unordered edge pair is visited once and contributes both orders: the high-order term is
 *    symmetric in (I, J), and the low-order increments have the same sign pattern for both
 *    orders, so only the weight `q_IJ + q_JI` differs;
 *  - `|T_I × (γ_i − γ_j)|` is the same for both endpoints `i` of edge `I` (they differ by a
 *    multiple of `T_I`), but it is still evaluated per endpoint, as the reference does, because
 *    the ε added after the norm makes the two differ at the 1e-10 level.
 * ε placement (after each norm, separately) follows the reference.
 * @see src/core/sobolev/innerProduct.ts (the per-pair formulas and their spec anchors)
 */
export function assembleScalarA(
    vertices: readonly Vec3[],
    edges: readonly Edge[],
    alpha: number,
    beta: number,
    epsilon: number,
): Float64Array {
    const n = vertices.length;
    const m = edges.length;
    const distExp = 2 * ((beta - 1) / alpha - 1) + 1;
    const X = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) {
        X[3 * i] = vertices[i][0];
        X[3 * i + 1] = vertices[i][1];
        X[3 * i + 2] = vertices[i][2];
    }
    // Vertex-pair tables: D = |γ_i − γ_j| + ε and P = D^−distExp (symmetric, zero diagonal).
    const D = new Float64Array(n * n);
    const P = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
        const xi = X[3 * i];
        const yi = X[3 * i + 1];
        const zi = X[3 * i + 2];
        for (let j = i + 1; j < n; j++) {
            const dx = xi - X[3 * j];
            const dy = yi - X[3 * j + 1];
            const dz = zi - X[3 * j + 2];
            const d = Math.sqrt(dx * dx + dy * dy + dz * dz) + epsilon;
            const p = 1 / d ** distExp;
            D[i * n + j] = d;
            D[j * n + i] = d;
            P[i * n + j] = p;
            P[j * n + i] = p;
        }
    }
    const e0 = new Int32Array(m);
    const e1 = new Int32Array(m);
    const ell = new Float64Array(m);
    const T = new Float64Array(3 * m);
    for (let I = 0; I < m; I++) {
        const a = edges[I][0];
        const b = edges[I][1];
        e0[I] = a;
        e1[I] = b;
        const ex = X[3 * b] - X[3 * a];
        const ey = X[3 * b + 1] - X[3 * a + 1];
        const ez = X[3 * b + 2] - X[3 * a + 2];
        const r = Math.sqrt(ex * ex + ey * ey + ez * ez);
        ell[I] = r + epsilon;
        // A degenerate edge has a zero unit tangent, as in the reference.
        if (r >= 1e-14) {
            T[3 * I] = ex / r;
            T[3 * I + 1] = ey / r;
            T[3 * I + 2] = ez / r;
        }
    }
    /** `(|T_I × (γ_i − γ_j)| + ε)² / D_ij⁴ · P_ij` — one term of the low-order kernel. */
    const low = (I: number, i: number, j: number): number => {
        const dx = X[3 * i] - X[3 * j];
        const dy = X[3 * i + 1] - X[3 * j + 1];
        const dz = X[3 * i + 2] - X[3 * j + 2];
        const tx = T[3 * I];
        const ty = T[3 * I + 1];
        const tz = T[3 * I + 2];
        const cx = ty * dz - tz * dy;
        const cy = tz * dx - tx * dz;
        const cz = tx * dy - ty * dx;
        const c = Math.sqrt(cx * cx + cy * cy + cz * cz) + epsilon;
        const d = D[i * n + j];
        const d2 = d * d;
        return ((c * c) / (d2 * d2)) * P[i * n + j];
    };

    const A = new Float64Array(n * n);
    for (let I = 0; I < m; I++) {
        const i0 = e0[I];
        const i1 = e1[I];
        const lI = ell[I];
        for (let J = I + 1; J < m; J++) {
            const j0 = e0[J];
            const j1 = e1[J];
            if (i0 === j0 || i0 === j1 || i1 === j0 || i1 === j1) continue;
            const lJ = ell[J];
            const p00 = P[i0 * n + j0];
            const p01 = P[i0 * n + j1];
            const p10 = P[i1 * n + j0];
            const p11 = P[i1 * n + j1];
            // High order. The reference visits (I,J) and (J,I); each visit adds the same four
            // increments, hence the factor 2 on w.
            const w = 2 * 0.25 * lI * lJ * (p00 + p01 + p10 + p11);
            const dotT =
                T[3 * I] * T[3 * J] + T[3 * I + 1] * T[3 * J + 1] + T[3 * I + 2] * T[3 * J + 2];
            const hI = w / (lI * lI);
            const hJ = w / (lJ * lJ);
            const hX = (w * dotT) / (lI * lJ);
            // Low order: q_IJ uses T_I, q_JI uses T_J; same increments for both orders.
            const q =
                0.0625 *
                lI *
                lJ *
                (low(I, i0, j0) +
                    low(I, i0, j1) +
                    low(I, i1, j0) +
                    low(I, i1, j1) +
                    low(J, j0, i0) +
                    low(J, j0, i1) +
                    low(J, j1, i0) +
                    low(J, j1, i1));
            // The reference's 2×2 loop over local endpoints (a, b) with sign (−1)^(a+b), written
            // out. Within an edge: + on the diagonal, − off it (high), + everywhere (low).
            A[i0 * n + i0] += hI + q;
            A[i1 * n + i1] += hI + q;
            A[i0 * n + i1] += q - hI;
            A[i1 * n + i0] += q - hI;
            A[j0 * n + j0] += hJ + q;
            A[j1 * n + j1] += hJ + q;
            A[j0 * n + j1] += q - hJ;
            A[j1 * n + j0] += q - hJ;
            // Across the two edges: high order carries the sign, low order is − everywhere.
            // (a, b) = (0,0),(1,1): sign +  →  −hX ;  (0,1),(1,0): sign −  →  +hX.
            A[i0 * n + j0] -= hX + q;
            A[j0 * n + i0] -= hX + q;
            A[i1 * n + j1] -= hX + q;
            A[j1 * n + i1] -= hX + q;
            A[i0 * n + j1] += hX - q;
            A[j1 * n + i0] += hX - q;
            A[i1 * n + j0] += hX - q;
            A[j0 * n + i1] += hX - q;
        }
    }
    return A;
}

/**
 * In-place Cholesky `M = L Lᵀ` of a symmetric positive definite row-major matrix; on return the
 * lower triangle holds `L` (the strict upper triangle is left as it was). Returns `false`, with
 * the matrix partly overwritten, when a pivot is not positive — the caller falls back.
 *
 * Why row-oriented (Cholesky–Banachiewicz): the inner loop is a dot product of two CONTIGUOUS
 * row prefixes, which is what a JS engine runs fastest; the four accumulators break the
 * dependency chain. Summation order therefore differs from a textbook loop — fine for a
 * factorization, never use this where bit-identity with `ldltFactor` matters.
 */
export function choleskyInPlace(M: Float64Array, n: number): boolean {
    for (let i = 0; i < n; i++) {
        const ri = i * n;
        for (let j = 0; j <= i; j++) {
            const rj = j * n;
            let s0 = 0;
            let s1 = 0;
            let s2 = 0;
            let s3 = 0;
            let k = 0;
            for (const kEnd = j - 3; k < kEnd; k += 4) {
                s0 += M[ri + k] * M[rj + k];
                s1 += M[ri + k + 1] * M[rj + k + 1];
                s2 += M[ri + k + 2] * M[rj + k + 2];
                s3 += M[ri + k + 3] * M[rj + k + 3];
            }
            for (; k < j; k++) s0 += M[ri + k] * M[rj + k];
            const v = M[ri + j] - (s0 + s1 + (s2 + s3));
            if (j < i) M[ri + j] = v / M[rj + j];
            else {
                if (!(v > 0)) return false;
                M[ri + i] = Math.sqrt(v);
            }
        }
    }
    return true;
}

/** What {@link makeDirectSolve} hands to `prepareSchur` / `prepareNullspace`. */
export interface DirectSolve {
    /** `applyA` is the exact dense `Ā u`; `sigma = tr(A)/n`. */
    op: HierarchicalOperator;
    /** `apply` is the exact `A_σ⁻¹`, block by block; `n` and `sigma` as the seam requires. */
    precond: { n: number; sigma: number; apply(R: ArrayLike<number>, out: Float64Array): void };
    /** Wall time of the assembly and of the factorization, for the timing ledger. */
    assembleMs: number;
    factorMs: number;
}

/**
 * Assemble, factor, and wrap. Returns `null` when the factorization meets a non-positive pivot
 * (a degenerate curve); the caller then uses another route.
 *
 * `A` has the constants in its kernel (translations), so it is `A_σ = A + (σ/n)·11ᵀ` that is
 * factored — the same shift, with the same σ = tr(A)/n, that the hierarchical route solves
 * with (`gmres.ts` `applyAsigma`). The operator handed out applies the UNSHIFTED `A`, because
 * that function adds the shift itself.
 */
export function makeDirectSolve(
    vertices: readonly Vec3[],
    edges: readonly Edge[],
    alpha: number,
    beta: number,
    epsilon: number,
): DirectSolve | null {
    const n = vertices.length;
    const t0 = performance.now();
    const A = assembleScalarA(vertices, edges, alpha, beta, epsilon);
    let trace = 0;
    for (let i = 0; i < n; i++) trace += A[i * n + i];
    const sigma = trace / n;
    const t1 = performance.now();
    const L = new Float64Array(n * n);
    const shift = sigma / n;
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) L[i * n + j] = A[i * n + j] + shift;
    if (!choleskyInPlace(L, n)) return null;
    const t2 = performance.now();

    const applyA = (u: ArrayLike<number>, out: Float64Array): void => {
        // The three coordinate blocks share one sweep over each row of A.
        for (let i = 0; i < n; i++) {
            const r = i * n;
            let s0 = 0;
            let s1 = 0;
            let s2 = 0;
            for (let j = 0; j < n; j++) {
                const a = A[r + j];
                s0 += a * u[j];
                s1 += a * u[n + j];
                s2 += a * u[2 * n + j];
            }
            out[i] = s0;
            out[n + i] = s1;
            out[2 * n + i] = s2;
        }
    };
    const apply = (R: ArrayLike<number>, out: Float64Array): void => {
        // L y = R (row-oriented), then Lᵀ x = y as a sweep that subtracts row i of L from the
        // unknowns above it — both read L along rows, never down a column.
        for (let i = 0; i < n; i++) {
            const r = i * n;
            let s0 = 0;
            let s1 = 0;
            let s2 = 0;
            for (let j = 0; j < i; j++) {
                const l = L[r + j];
                s0 += l * out[j];
                s1 += l * out[n + j];
                s2 += l * out[2 * n + j];
            }
            const d = L[r + i];
            out[i] = (R[i] - s0) / d;
            out[n + i] = (R[n + i] - s1) / d;
            out[2 * n + i] = (R[2 * n + i] - s2) / d;
        }
        for (let i = n - 1; i >= 0; i--) {
            const r = i * n;
            const d = L[r + i];
            const x0 = out[i] / d;
            const x1 = out[n + i] / d;
            const x2 = out[2 * n + i] / d;
            out[i] = x0;
            out[n + i] = x1;
            out[2 * n + i] = x2;
            for (let j = 0; j < i; j++) {
                const l = L[r + j];
                out[j] -= l * x0;
                out[n + j] -= l * x1;
                out[2 * n + j] -= l * x2;
            }
        }
    };
    return {
        op: { n, sigma, applyA },
        precond: { n, sigma, apply },
        assembleMs: t1 - t0,
        factorMs: t2 - t1,
    };
}
