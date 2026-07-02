/**
 * Dense linear solver + saddle-system assembly for the fractional
 * Sobolev-gradient solve (Repulsive Curves, Yu/Schumacher/Crane 2021).
 *
 * Hand-rolled on purpose, no dependency: every solve returns the relative
 * residual ‖K·z − rhs‖₂ / max(1, ‖rhs‖₂), so correctness is self-certifying at
 * runtime — that machine-checked residual is the reason we do not import a
 * linear-algebra package. LU with partial pivoting is the spec's sanctioned
 * dense fallback for the symmetric-indefinite saddle matrix (the preferred
 * Bunch–Kaufman LDLᵀ is deliberately not required — reviewability first).
 * Do NOT use Cholesky on the saddle matrix (it is indefinite).
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Gradient saddle system" — solver list)
 * @see oracle/tpe_stage1_oracle.py (solve_saddle)
 */

/**
 * Solves K·x = rhs by dense LU factorization with partial (row) pivoting,
 * followed by forward/back substitution. Plain loops, no dependency.
 *
 * Operates on copies — never mutates `K` or `rhs`. Throws an Error mentioning
 * "singular" when a pivot column's max abs entry is 0 or any input entry is
 * non-finite (a non-finite matrix cannot be certified by the residual, so it
 * is rejected up front rather than propagated as NaN).
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Acceptable dense fallback: LU with partial pivoting")
 */
export function luSolve(K: number[][], rhs: number[]): number[] {
    const n = K.length;
    if (rhs.length !== n) {
        throw new Error(`luSolve: rhs length ${rhs.length} does not match matrix size ${n}`);
    }
    // Copy — never mutate the caller's matrix/rhs. The factorization below is
    // in-place on M (L multipliers in the strict lower triangle, U on and above
    // the diagonal), so working on the original would corrupt it.
    const M: number[][] = new Array(n);
    for (let i = 0; i < n; i++) {
        const row = K[i];
        if (row.length !== n) {
            throw new Error(`luSolve: matrix row ${i} has length ${row.length}, expected ${n}`);
        }
        for (let j = 0; j < n; j++) {
            if (!Number.isFinite(row[j])) {
                throw new Error(
                    `luSolve: non-finite matrix entry at (${i},${j}) — treating as singular`,
                );
            }
        }
        M[i] = row.slice();
    }
    for (let i = 0; i < n; i++) {
        if (!Number.isFinite(rhs[i])) {
            throw new Error(`luSolve: non-finite rhs entry at ${i} — treating as singular`);
        }
    }

    // piv[i] = original row index now living at row i (rows of M are swapped
    // physically; piv tracks the permutation so P·rhs can be applied later).
    const piv = Array.from({ length: n }, (_, i) => i);

    for (let col = 0; col < n; col++) {
        // Partial pivoting: pick the row with the largest |entry| in this column.
        let pivotRow = col;
        let maxAbs = Math.abs(M[col][col]);
        for (let r = col + 1; r < n; r++) {
            const a = Math.abs(M[r][col]);
            if (a > maxAbs) {
                maxAbs = a;
                pivotRow = r;
            }
        }
        // `!(maxAbs > 0)` also catches NaN, as a second line of defense behind
        // the up-front finiteness scan.
        if (!(maxAbs > 0)) {
            throw new Error(
                `luSolve: matrix is singular (pivot column ${col} has max abs entry 0)`,
            );
        }
        if (pivotRow !== col) {
            const tmpRow = M[col];
            M[col] = M[pivotRow];
            M[pivotRow] = tmpRow;
            const tmpPiv = piv[col];
            piv[col] = piv[pivotRow];
            piv[pivotRow] = tmpPiv;
        }
        const pivotVal = M[col][col];
        for (let r = col + 1; r < n; r++) {
            const m = M[r][col] / pivotVal;
            M[r][col] = m; // store the L multiplier in the eliminated slot
            for (let c = col + 1; c < n; c++) {
                M[r][c] -= m * M[col][c];
            }
        }
    }

    // Forward substitution: L·y = P·rhs (L unit lower triangular).
    const y = new Array<number>(n);
    for (let i = 0; i < n; i++) {
        let s = rhs[piv[i]];
        for (let j = 0; j < i; j++) {
            s -= M[i][j] * y[j];
        }
        y[i] = s;
    }
    // Back substitution: U·x = y.
    const x = new Array<number>(n);
    for (let i = n - 1; i >= 0; i--) {
        let s = y[i];
        for (let j = i + 1; j < n; j++) {
            s -= M[i][j] * x[j];
        }
        x[i] = s / M[i][i];
    }
    return x;
}

/**
 * Assembles the (3n+k)×(3n+k) symmetric-indefinite saddle matrix
 * `K = [[Ā, Cᵀ], [C, 0]]` from the 3n×3n block-diagonal inner-product matrix
 * Ā (= A3, see `expandBlockDiag` in `./layout`) and the k×3n constraint
 * Jacobian C. k = 3 for the barycenter constraint today, but the assembly is
 * written for any k — more constraint rows (e.g. edge lengths) are appended
 * in stage 2.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Gradient saddle system")
 * @see oracle/tpe_stage1_oracle.py (solve_saddle — np.block([[A3, C.T], [C, 0]]))
 */
export function buildSaddleMatrix(A3: number[][], C: number[][]): number[][] {
    const m = A3.length; // 3n
    const k = C.length;
    const size = m + k;
    const K: number[][] = Array.from({ length: size }, () => new Array<number>(size).fill(0));
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < m; j++) {
            K[i][j] = A3[i][j];
        }
    }
    for (let r = 0; r < k; r++) {
        for (let j = 0; j < m; j++) {
            K[m + r][j] = C[r][j]; // C block
            K[j][m + r] = C[r][j]; // Cᵀ block
        }
    }
    // Lower-right k×k block stays exactly 0 (from the fill above): the saddle
    // system has no constraint-constraint coupling. Do NOT add a regularizing
    // identity here — that changes the metric.
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Do not 'fix' singularity by adding a large identity")
    return K;
}

/**
 * Solves the constrained Sobolev-gradient saddle system
 * `[[Ā, Cᵀ], [C, 0]]·[x; λ] = [rhsTop; rhsBottom]` and splits the solution.
 *
 * `rhsBottom` defaults to zeros(k) — the gradient solve's bottom block is 0;
 * the constraint-projection solve passes −Φ instead (same system, stage-2).
 * The relative residual ‖K·z − rhs‖₂ / max(1, ‖rhs‖₂) is ALWAYS computed and
 * returned: it is the self-certifying correctness check for the hand-rolled
 * solver (spec §E prop 8 gates it at ≤1e-10), and its O(N²) cost is negligible
 * next to the O(N³) factorization. Mirrors the oracle's solve_saddle semantics
 * (same residual definition).
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Gradient saddle system")
 * @see oracle/tpe_stage1_oracle.py (solve_saddle)
 */
export function solveSaddle(
    A3: number[][],
    C: number[][],
    rhsTop: number[],
    rhsBottom?: number[],
): { x: number[]; lambda: number[]; residual: number } {
    const m = A3.length; // 3n
    const k = C.length;
    if (rhsTop.length !== m) {
        throw new Error(`solveSaddle: rhsTop length ${rhsTop.length} does not match A3 size ${m}`);
    }
    const bottom = rhsBottom ?? new Array<number>(k).fill(0);
    if (bottom.length !== k) {
        throw new Error(
            `solveSaddle: rhsBottom length ${bottom.length} does not match C rows ${k}`,
        );
    }
    const K = buildSaddleMatrix(A3, C);
    const rhs = [...rhsTop, ...bottom];
    const z = luSolve(K, rhs);

    // Self-certifying residual — computed on EVERY solve, never skipped.
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B / §E (prop 8: saddle residual)
    let resNormSq = 0;
    let rhsNormSq = 0;
    const size = m + k;
    for (let i = 0; i < size; i++) {
        const Ki = K[i];
        let s = 0;
        for (let j = 0; j < size; j++) {
            s += Ki[j] * z[j];
        }
        const d = s - rhs[i];
        resNormSq += d * d;
        rhsNormSq += rhs[i] * rhs[i];
    }
    const residual = Math.sqrt(resNormSq) / Math.max(1, Math.sqrt(rhsNormSq));

    return { x: z.slice(0, m), lambda: z.slice(m), residual };
}
