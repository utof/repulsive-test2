/**
 * Dense linear solver + saddle-system assembly for the fractional
 * Sobolev-gradient solve (Repulsive Curves, Yu/Schumacher/Crane 2021).
 *
 * Hand-rolled on purpose, no dependency: every solve returns the relative
 * residual ‖K·z − rhs‖₂ / max(1, ‖rhs‖₂), so correctness is self-certifying at
 * runtime — that machine-checked residual is the reason we do not import a
 * linear-algebra package. This holds for BOTH paths in this module: the
 * `number[][]` reference path (`luSolve`/`solveSaddle`, golden-tested) and the
 * flat typed-array fast path (`luFactor`/`luSolveFactored`/`solveSaddleFromA`,
 * solver-perf Task 5) — the fast path computes the residual on EVERY solve via
 * a structured matvec against the original A and C, never skipped.
 * LU with partial pivoting is the spec's sanctioned dense fallback for the
 * symmetric-indefinite saddle matrix (the preferred Bunch–Kaufman LDLᵀ is
 * deliberately not required — reviewability first).
 * Do NOT use Cholesky on the saddle matrix (it is indefinite).
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Gradient saddle system" — solver list)
 * @see oracle/tpe_stage1_oracle.py (solve_saddle)
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 5)
 */
import { timed } from './phaseTimings';

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

/**
 * A dense LU factorization with partial pivoting, packed for reuse
 * ({@link luSolveFactored}; frozen-projection reuse consumes it in Task 6).
 * `m` holds L (strict lower triangle, unit diagonal implied) and U (on and
 * above the diagonal) in one flat row-major n×n buffer; `piv[i]` is the
 * original row index now living at row i (rows were swapped physically).
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 5)
 */
export interface LuFactorization {
    m: Float64Array;
    piv: Int32Array;
    n: number;
}

/**
 * Factors a flat row-major n×n matrix by dense LU with partial (row) pivoting
 * — the same algorithm, pivot rule, and op order as {@link luSolve}'s
 * factorization stage, so a factor+solve round trip is bit-identical to
 * luSolve on the same input.
 *
 * WARNING: factors IN PLACE on the caller's buffer `k` — the buffer is
 * consumed (it becomes the packed L/U storage, or garbage if this throws
 * mid-factorization). Never pass a buffer you still need; this is the whole
 * point of the flat fast path (no per-solve row copies, briefing §2.1).
 *
 * Same two throw contracts as luSolve: any non-finite entry → Error mentioning
 * "treating as singular" (scanned up front — a non-finite matrix cannot be
 * certified by the residual); a pivot column whose max abs entry is 0 →
 * Error mentioning "singular".
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Acceptable dense fallback: LU with partial pivoting")
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 5)
 */
export function luFactor(k: Float64Array, n: number): LuFactorization {
    if (k.length !== n * n) {
        throw new Error(`luFactor: buffer length ${k.length} does not match n*n = ${n * n}`);
    }
    for (let idx = 0; idx < n * n; idx++) {
        if (!Number.isFinite(k[idx])) {
            throw new Error(
                `luFactor: non-finite matrix entry at (${Math.floor(idx / n)},${idx % n}) — treating as singular`,
            );
        }
    }

    // piv[i] = original row index now living at row i — same permutation
    // bookkeeping as luSolve, so P·rhs can be applied in luSolveFactored.
    const piv = new Int32Array(n);
    for (let i = 0; i < n; i++) piv[i] = i;

    for (let col = 0; col < n; col++) {
        // Partial pivoting: pick the row with the largest |entry| in this column.
        let pivotRow = col;
        let maxAbs = Math.abs(k[col * n + col]);
        for (let r = col + 1; r < n; r++) {
            const a = Math.abs(k[r * n + col]);
            if (a > maxAbs) {
                maxAbs = a;
                pivotRow = r;
            }
        }
        // `!(maxAbs > 0)` also catches NaN, as a second line of defense behind
        // the up-front finiteness scan (same guard as luSolve).
        if (!(maxAbs > 0)) {
            throw new Error(
                `luFactor: matrix is singular (pivot column ${col} has max abs entry 0)`,
            );
        }
        if (pivotRow !== col) {
            // Element-wise row swap — a flat buffer has no row pointers to swap,
            // but the values (and thus all downstream arithmetic) are identical
            // to luSolve's pointer swap.
            const a = col * n;
            const b = pivotRow * n;
            for (let c = 0; c < n; c++) {
                const t = k[a + c];
                k[a + c] = k[b + c];
                k[b + c] = t;
            }
            const tmpPiv = piv[col];
            piv[col] = piv[pivotRow];
            piv[pivotRow] = tmpPiv;
        }
        const colBase = col * n;
        const pivotVal = k[colBase + col];
        for (let r = col + 1; r < n; r++) {
            const rBase = r * n;
            const mult = k[rBase + col] / pivotVal;
            k[rBase + col] = mult; // store the L multiplier in the eliminated slot
            // Pointer-bump indexing instead of `k[rBase + c]` — measured ~1.4×
            // faster in Bun/JSC on the O(n³) inner loop (JSC does not strength-
            // reduce the computed index); op order is unchanged (c ascending),
            // so results stay bit-identical to luSolve.
            // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 5, step 5.5 measurement note)
            let dst = rBase + col + 1;
            let src = colBase + col + 1;
            const end = rBase + n;
            while (dst < end) {
                k[dst] -= mult * k[src];
                dst++;
                src++;
            }
        }
    }

    return { m: k, piv, n };
}

/**
 * Forward/back substitution against a {@link luFactor} result — the
 * substitution stage of {@link luSolve}, op-for-op, on the packed flat
 * factors. Never mutates the factorization or `rhs`; keeps luSolve's
 * non-finite-rhs throw contract (Error mentioning "treating as singular").
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Acceptable dense fallback: LU with partial pivoting")
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 5)
 */
export function luSolveFactored(fac: LuFactorization, rhs: number[]): number[] {
    const { m, piv, n } = fac;
    if (rhs.length !== n) {
        throw new Error(
            `luSolveFactored: rhs length ${rhs.length} does not match matrix size ${n}`,
        );
    }
    for (let i = 0; i < n; i++) {
        if (!Number.isFinite(rhs[i])) {
            throw new Error(`luSolveFactored: non-finite rhs entry at ${i} — treating as singular`);
        }
    }

    // Forward substitution: L·y = P·rhs (L unit lower triangular).
    const y = new Array<number>(n);
    for (let i = 0; i < n; i++) {
        let s = rhs[piv[i]];
        const base = i * n;
        for (let j = 0; j < i; j++) {
            s -= m[base + j] * y[j];
        }
        y[i] = s;
    }
    // Back substitution: U·x = y.
    const x = new Array<number>(n);
    for (let i = n - 1; i >= 0; i--) {
        let s = y[i];
        const base = i * n;
        for (let j = i + 1; j < n; j++) {
            s -= m[base + j] * x[j];
        }
        x[i] = s / m[base + i];
    }
    return x;
}

/**
 * Factor-selection switch for the typed-array saddle fast path (LDLᵀ A/B).
 * 'lu': dense LU with partial pivoting ({@link luFactor}) — the default and
 * the semantics of ALL committed goldens. 'ldlt': symmetric-indefinite
 * Bunch–Kaufman LDLᵀ ({@link ldltFactor}) — ~half the factor flops on the
 * same flat buffer; gated by the pre-registered A/B before any default flip.
 * @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (pinned decision 4)
 */
export type FactorMode = 'lu' | 'ldlt';

/**
 * A symmetric-indefinite Bunch–Kaufman LDLᵀ factorization, packed LAPACK
 * `dsytf2`-style in the same flat row-major buffer convention as
 * {@link LuFactorization}: `m` holds the multiplier columns of the permuted
 * unit lower-triangular L plus the block-diagonal D (diagonal entries, and
 * the subdiagonal entry of each 2×2 pivot block); everything strictly above
 * the diagonal is dead storage. `ipiv` keeps LAPACK's 1-BASED encoding so the
 * sign test works at index 0: `ipiv[k] = kp+1` → 1×1 pivot, rows/cols k↔kp
 * interchanged; `ipiv[k] = ipiv[k+1] = −(kp+1)` → 2×2 pivot, rows/cols
 * (k+1)↔kp interchanged. `kind` discriminates from {@link LuFactorization}
 * (which predates the union and stays field-compatible with its consumers).
 * @see https://raw.githubusercontent.com/Reference-LAPACK/lapack/master/SRC/dsytf2.f (UPLO='L')
 * @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (pinned decision 1)
 */
export interface LdltFactorization {
    kind: 'ldlt';
    m: Float64Array;
    ipiv: Int32Array;
    n: number;
}

/**
 * Either dense factorization the saddle fast path can produce/consume —
 * selected by {@link FactorMode}, dispatched by the `kind` discriminant
 * (absent on the LU shape).
 * @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (pinned decision 4)
 */
export type SaddleFactorization = LuFactorization | LdltFactorization;

/**
 * Factors a flat row-major SYMMETRIC n×n matrix as P·L·D·Lᵀ·Pᵀ by
 * Bunch–Kaufman partial pivoting (1×1/2×2 pivots) — an unblocked port of
 * LAPACK `dsytf2` (UPLO='L'), verified against the reference source
 * (pivot threshold α = (1+√17)/8, the three-way pivot test, the 2×2 update
 * formulas, and the ipiv encoding are LAPACK's verbatim). Only the LOWER
 * triangle is read and written; ~n³/6 multiply-adds vs LU's ~n³/3 — that
 * halving is the whole point of the A/B (plan 2026-07-06).
 *
 * LAPACK is column-major; this buffer is row-major, so the trailing updates
 * are restructured row-wise with the per-column coefficients precomputed into
 * O(n) scratch — numerically EXACT w.r.t. dsytf2: each trailing element
 * receives exactly one update expression per elimination step and the
 * expression (operand order included) is LAPACK's; only the element visit
 * order changes, which cannot change any element's value.
 * @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (pinned decision 1)
 *
 * WARNING: factors IN PLACE on the caller's buffer `k` — consumed, exactly
 * like {@link luFactor} (same rationale: no per-solve copies on the fast path).
 *
 * Same two throw contracts as {@link luFactor} (pinned decision 3 — the
 * existing singular contract, never a new throw type): any non-finite entry →
 * Error mentioning "treating as singular" (scanned up front); an exactly-zero
 * pivot column (max(|A(k,k)|, colmax) = 0) → Error mentioning "singular".
 * Where LAPACK would set INFO>0 and emit a zero D block (making the
 * subsequent solve divide by zero), we throw — callers already catch and fold
 * into 'singular_system' / ok:false (optimizer.ts, lineSearch.ts).
 * @see https://raw.githubusercontent.com/Reference-LAPACK/lapack/master/SRC/dsytf2.f
 * @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (pinned decisions 1, 3)
 */
export function ldltFactor(k: Float64Array, n: number): LdltFactorization {
    if (k.length !== n * n) {
        throw new Error(`ldltFactor: buffer length ${k.length} does not match n*n = ${n * n}`);
    }
    for (let idx = 0; idx < n * n; idx++) {
        if (!Number.isFinite(k[idx])) {
            throw new Error(
                `ldltFactor: non-finite matrix entry at (${Math.floor(idx / n)},${idx % n}) — treating as singular`,
            );
        }
    }

    // Bunch–Kaufman growth-optimal pivot threshold (dsytf2: ALPHA = (ONE+SQRT(SEVTEN))/EIGHT).
    const ALPHA = (1 + Math.sqrt(17)) / 8;
    const ipiv = new Int32Array(n);
    // Row-major restructure scratch: per-column update coefficients (w1 = the
    // 1×1 d11·column / 2×2 WK column, w2 = the 2×2 WKP1 column).
    const w1 = new Float64Array(n);
    const w2 = new Float64Array(n);

    let col = 0;
    while (col < n) {
        const colBase = col * n;
        const absakk = Math.abs(k[colBase + col]);

        // imax = row of the largest |subdiagonal| entry in column col (first
        // max wins, matching IDAMAX); colmax = that magnitude.
        let imax = col;
        let colmax = 0;
        for (let r = col + 1; r < n; r++) {
            const a = Math.abs(k[r * n + col]);
            if (a > colmax) {
                colmax = a;
                imax = r;
            }
        }

        // `!(… > 0)` also catches NaN, second line of defense behind the
        // up-front scan (same guard shape as luFactor). LAPACK sets INFO and
        // continues; we throw per the pinned breakdown contract.
        if (!(Math.max(absakk, colmax) > 0)) {
            throw new Error(
                `ldltFactor: matrix is singular (pivot column ${col} has max abs entry 0)`,
            );
        }

        // Three-way Bunch–Kaufman pivot decision — dsytf2's exact inequality forms.
        let kstep = 1;
        let kp = col;
        if (absakk >= ALPHA * colmax) {
            kp = col; // 1×1, no interchange
        } else {
            // rowmax = largest |off-diagonal| in row/col imax of the trailing
            // submatrix: row part A(imax, col..imax-1), column part A(imax+1.., imax).
            let rowmax = 0;
            const imaxBase = imax * n;
            for (let j = col; j < imax; j++) {
                rowmax = Math.max(rowmax, Math.abs(k[imaxBase + j]));
            }
            for (let r = imax + 1; r < n; r++) {
                rowmax = Math.max(rowmax, Math.abs(k[r * n + imax]));
            }
            if (absakk >= ALPHA * colmax * (colmax / rowmax)) {
                kp = col; // 1×1, no interchange
            } else if (Math.abs(k[imaxBase + imax]) >= ALPHA * rowmax) {
                kp = imax; // 1×1, interchange col↔imax
            } else {
                kp = imax; // 2×2, interchange (col+1)↔imax
                kstep = 2;
            }
        }

        // Symmetric interchange of rows/cols kk↔kp in the TRAILING submatrix
        // only (columns < col of packed L are NOT re-permuted — dsytf2's
        // scheme; the solve applies interchanges interleaved to compensate).
        const kk = col + kstep - 1;
        if (kp !== kk) {
            for (let r = kp + 1; r < n; r++) {
                const t = k[r * n + kk];
                k[r * n + kk] = k[r * n + kp];
                k[r * n + kp] = t;
            }
            for (let j = kk + 1; j < kp; j++) {
                const t = k[j * n + kk];
                k[j * n + kk] = k[kp * n + j];
                k[kp * n + j] = t;
            }
            const td = k[kk * n + kk];
            k[kk * n + kk] = k[kp * n + kp];
            k[kp * n + kp] = td;
            if (kstep === 2) {
                const t = k[(col + 1) * n + col];
                k[(col + 1) * n + col] = k[kp * n + col];
                k[kp * n + col] = t;
            }
        }

        if (kstep === 1) {
            // Rank-1 update of the trailing lower triangle:
            // A(i,j) -= A(i,k)·(d11·A(j,k)), then scale the column by d11
            // (dsytf2's DSYR + DSCAL, row-wise visit order).
            if (col < n - 1) {
                const d11 = 1 / k[colBase + col];
                for (let j = col + 1; j < n; j++) w1[j] = d11 * k[j * n + col];
                for (let i = col + 1; i < n; i++) {
                    const iBase = i * n;
                    const aik = k[iBase + col];
                    for (let j = col + 1; j <= i; j++) {
                        k[iBase + j] -= aik * w1[j];
                    }
                }
                for (let r = col + 1; r < n; r++) k[r * n + col] *= d11;
            }
        } else {
            // Rank-2 update — dsytf2's 2×2 formulas verbatim. d21 ≠ 0 is
            // guaranteed: after the interchange |A(col+1,col)| = colmax > 0.
            if (col < n - 2) {
                const d21 = k[(col + 1) * n + col];
                const d11 = k[(col + 1) * n + col + 1] / d21;
                const d22 = k[colBase + col] / d21;
                const t = 1 / (d11 * d22 - 1);
                const d21t = t / d21;
                for (let j = col + 2; j < n; j++) {
                    const ajk = k[j * n + col];
                    const ajk1 = k[j * n + col + 1];
                    w1[j] = d21t * (d11 * ajk - ajk1); // WK
                    w2[j] = d21t * (d22 * ajk1 - ajk); // WKP1
                }
                for (let i = col + 2; i < n; i++) {
                    const iBase = i * n;
                    const aik = k[iBase + col];
                    const aik1 = k[iBase + col + 1];
                    for (let j = col + 2; j <= i; j++) {
                        // Two separate subtractions — dsytf2's exact expression
                        // A(I,J) = A(I,J) - A(I,K)*WK - A(I,K+1)*WKP1.
                        k[iBase + j] = k[iBase + j] - aik * w1[j] - aik1 * w2[j];
                    }
                }
                // Store the L multipliers AFTER the full trailing update — the
                // update above reads the pre-overwrite A(i,col) values, exactly
                // the values dsytf2's column-ascending order would read.
                for (let j = col + 2; j < n; j++) {
                    k[j * n + col] = w1[j];
                    k[j * n + col + 1] = w2[j];
                }
            }
        }

        if (kstep === 1) {
            ipiv[col] = kp + 1;
        } else {
            ipiv[col] = -(kp + 1);
            ipiv[col + 1] = -(kp + 1);
        }
        col += kstep;
    }

    return { kind: 'ldlt', m: k, ipiv, n };
}

/**
 * Solves against a {@link ldltFactor} result — a single-RHS port of LAPACK
 * `dsytrs` (UPLO='L'), verified against the reference source: forward sweep
 * applying the interleaved interchanges + inv(L) + the block-diagonal inverse
 * (2×2 blocks via dsytrs's exact scaled-determinant formula), then the
 * backward inv(Lᵀ) sweep with the interchanges re-applied in reverse.
 * Never mutates the factorization or `rhs`; keeps the non-finite-rhs throw
 * contract of {@link luSolveFactored} (Error mentioning "treating as
 * singular").
 * @see https://raw.githubusercontent.com/Reference-LAPACK/lapack/master/SRC/dsytrs.f (UPLO='L')
 * @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (pinned decision 1)
 */
export function ldltSolveFactored(fac: LdltFactorization, rhs: number[]): number[] {
    const { m, ipiv, n } = fac;
    if (rhs.length !== n) {
        throw new Error(
            `ldltSolveFactored: rhs length ${rhs.length} does not match matrix size ${n}`,
        );
    }
    for (let i = 0; i < n; i++) {
        if (!Number.isFinite(rhs[i])) {
            throw new Error(
                `ldltSolveFactored: non-finite rhs entry at ${i} — treating as singular`,
            );
        }
    }

    const b = rhs.slice();

    // Solve L·D·y = b (forward, interchanges interleaved — dsytrs 'L' first loop).
    let i = 0;
    while (i < n) {
        if (ipiv[i] > 0) {
            const kp = ipiv[i] - 1;
            if (kp !== i) {
                const t = b[i];
                b[i] = b[kp];
                b[kp] = t;
            }
            for (let r = i + 1; r < n; r++) b[r] -= m[r * n + i] * b[i];
            b[i] /= m[i * n + i];
            i += 1;
        } else {
            // 2×2 block at (i, i+1); only row i+1 was interchanged at factor time.
            const kp = -ipiv[i] - 1;
            if (kp !== i + 1) {
                const t = b[i + 1];
                b[i + 1] = b[kp];
                b[kp] = t;
            }
            for (let r = i + 2; r < n; r++) {
                b[r] = b[r] - m[r * n + i] * b[i] - m[r * n + i + 1] * b[i + 1];
            }
            // dsytrs's 2×2 diagonal-block solve, verbatim (scaled by the
            // subdiagonal to avoid overflow in the determinant).
            const akm1k = m[(i + 1) * n + i];
            const akm1 = m[i * n + i] / akm1k;
            const ak = m[(i + 1) * n + i + 1] / akm1k;
            const denom = akm1 * ak - 1;
            const bkm1 = b[i] / akm1k;
            const bk = b[i + 1] / akm1k;
            b[i] = (ak * bkm1 - bk) / denom;
            b[i + 1] = (akm1 * bk - bkm1) / denom;
            i += 2;
        }
    }

    // Solve Lᵀ·x = y (backward, interchanges re-applied in reverse —
    // dsytrs 'L' second loop; a 2×2 block is entered at its HIGHER index).
    i = n - 1;
    while (i >= 0) {
        if (ipiv[i] > 0) {
            let s = b[i];
            for (let r = i + 1; r < n; r++) s -= m[r * n + i] * b[r];
            b[i] = s;
            const kp = ipiv[i] - 1;
            if (kp !== i) {
                const t = b[i];
                b[i] = b[kp];
                b[kp] = t;
            }
            i -= 1;
        } else {
            let s = b[i];
            for (let r = i + 1; r < n; r++) s -= m[r * n + i] * b[r];
            b[i] = s;
            let s1 = b[i - 1];
            for (let r = i + 1; r < n; r++) s1 -= m[r * n + (i - 1)] * b[r];
            b[i - 1] = s1;
            const kp = -ipiv[i] - 1;
            if (kp !== i) {
                const t = b[i];
                b[i] = b[kp];
                b[kp] = t;
            }
            i -= 2;
        }
    }

    return b;
}

// Substitution dispatch on the factorization kind ('kind' exists only on the
// LDLᵀ shape). The 'lu' branch is the verbatim pre-existing call — default
// path stays bit-identical. @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (decision 4)
function solveFactored(fac: SaddleFactorization, rhs: number[]): number[] {
    return 'kind' in fac ? ldltSolveFactored(fac, rhs) : luSolveFactored(fac, rhs);
}

/**
 * Fast-path saddle solve from the SCALAR |V|×|V| inner-product matrix `a`
 * (flat row-major, from `assembleAFlat` in `./innerProduct`): builds the flat
 * (3n+k)×(3n+k) `K = [[Ā, Cᵀ], [C, 0]]` directly — Ā's three diagonal
 * coordinate blocks are written straight from `a`, `expandBlockDiag` is never
 * materialized — then factors ({@link luFactor}, wrapped in the 'factor'
 * timing phase) and solves. Numerically bit-identical to
 * `solveSaddle(expandBlockDiag(A), C, …)`: same values in K, same pivoting.
 *
 * The k×k lower-right block stays exactly 0: the saddle system has no
 * constraint-constraint coupling. Do NOT add a regularizing identity — that
 * changes the metric.
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Do not 'fix' singularity by adding a large identity")
 *
 * `rhsBottom` defaults to zeros(k) — the gradient solve's bottom block is 0;
 * the constraint-projection solve passes −Φ instead. k = 0 (EMPTY constraint
 * set, spec §9a) degenerates to the pure Ā·x = rhsTop solve, exactly like
 * solveSaddle.
 *
 * The relative residual ‖K·z − rhs‖₂ / max(1, ‖rhs‖₂) is ALWAYS computed and
 * returned — the self-certifying correctness check for the hand-rolled solver
 * (spec §E prop 8 gates it at ≤1e-10), same definition as solveSaddle and the
 * oracle's solve_saddle. Because the in-place factorization destroyed K, the
 * residual matvec is STRUCTURED against the original `a` and `C`:
 * K·z = [A·x per coordinate block + Cᵀ·λ ; C·x] — term order matches the dense
 * matvec's nonzero terms, so the value is unchanged.
 *
 * Returns `fac` so a caller can reuse the factorization for further solves of
 * the SAME frozen system (frozen-projection reuse, plan Task 6).
 *
 * `factorMode` selects the dense factorization (LDLᵀ A/B, plan 2026-07-06):
 * the default 'lu' path is bit-identical to the pre-option code; 'ldlt' runs
 * Bunch–Kaufman on the same K buffer and is gated by the SAME structured
 * residual — the self-certification never depends on the factorization.
 * @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (pinned decision 4)
 * @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Gradient saddle system")
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 5)
 * @see local_files/2026-07-03-next-steps-briefing.md §5A item 2
 */
export function solveSaddleFromA(
    a: Float64Array,
    n: number,
    C: number[][],
    rhsTop: number[],
    rhsBottom?: number[],
    factorMode: FactorMode = 'lu',
): { x: number[]; lambda: number[]; residual: number; fac: SaddleFactorization } {
    if (a.length !== n * n) {
        throw new Error(`solveSaddleFromA: a length ${a.length} does not match n*n = ${n * n}`);
    }
    const m = 3 * n;
    const k = C.length;
    if (rhsTop.length !== m) {
        throw new Error(
            `solveSaddleFromA: rhsTop length ${rhsTop.length} does not match 3n = ${m}`,
        );
    }
    const bottom = rhsBottom ?? new Array<number>(k).fill(0);
    if (bottom.length !== k) {
        throw new Error(
            `solveSaddleFromA: rhsBottom length ${bottom.length} does not match C rows ${k}`,
        );
    }

    const size = m + k;
    const K = new Float64Array(size * size); // zero-initialized: off-diagonal Ā blocks + k×k block stay 0
    for (let b = 0; b < 3; b++) {
        const off = b * n;
        for (let i = 0; i < n; i++) {
            const src = i * n;
            const dst = (off + i) * size + off;
            for (let j = 0; j < n; j++) {
                K[dst + j] = a[src + j];
            }
        }
    }
    for (let r = 0; r < k; r++) {
        const Cr = C[r];
        const rowBase = (m + r) * size;
        for (let j = 0; j < m; j++) {
            K[rowBase + j] = Cr[j]; // C block
            K[j * size + (m + r)] = Cr[j]; // Cᵀ block
        }
    }
    const rhs = [...rhsTop, ...bottom];

    // 'factor' timing sub-phase overlaps the caller's 'saddle' wrap by design
    // (see the phase-key schema note in ./phaseTimings).
    // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 5)
    const fac = timed('factor', () =>
        factorMode === 'ldlt' ? ldltFactor(K, size) : luFactor(K, size),
    );
    const z = solveFactored(fac, rhs);
    const residual = structuredSaddleResidual(a, n, C, z, rhs);

    return { x: z.slice(0, m), lambda: z.slice(m), residual, fac };
}

// Self-certifying residual ‖K·z − rhs‖₂ / max(1, ‖rhs‖₂) via a STRUCTURED
// matvec against the original a/C (the in-place factorization destroyed K):
// K·z = [A·x per coordinate block + Cᵀ·λ ; C·x]. Extracted verbatim from
// solveSaddleFromA (loop and term order unchanged — the golden-gated value is
// bit-identical) so solveSaddleFrozen certifies its reuse solves the same way.
function structuredSaddleResidual(
    a: Float64Array,
    n: number,
    C: number[][],
    z: number[],
    rhs: number[],
): number {
    const m = 3 * n;
    const k = C.length;
    let resNormSq = 0;
    let rhsNormSq = 0;
    for (let b = 0; b < 3; b++) {
        const off = b * n;
        for (let i = 0; i < n; i++) {
            let s = 0;
            const base = i * n;
            for (let j = 0; j < n; j++) {
                s += a[base + j] * z[off + j];
            }
            const row = off + i;
            for (let r = 0; r < k; r++) {
                s += C[r][row] * z[m + r];
            }
            const d = s - rhs[row];
            resNormSq += d * d;
            rhsNormSq += rhs[row] * rhs[row];
        }
    }
    for (let r = 0; r < k; r++) {
        const Cr = C[r];
        let s = 0;
        for (let j = 0; j < m; j++) {
            s += Cr[j] * z[j];
        }
        const d = s - rhs[m + r];
        resNormSq += d * d;
        rhsNormSq += rhs[m + r] * rhs[m + r];
    }
    return Math.sqrt(resNormSq) / Math.max(1, Math.sqrt(rhsNormSq));
}

/**
 * The frozen saddle operator K(γ₀) = [[Ā(γ₀), C(γ₀)ᵀ], [C(γ₀), 0]], packed for
 * per-step factorization reuse (solver-perf Task 6): `fac` is the one LU that
 * the gradient solve and every projection iterate of every τ-trial share —
 * the authors' reference-implementation scheme (ythea/repulsive-curves
 * src/tpe_flow_sc.cpp, ProjectGradient + LSBackproject; paper line 734).
 * `a` (flat scalar |V|×|V| Ā) and `C` are retained UNfactored so every reuse
 * solve can compute the same self-certifying structured residual as
 * solveSaddleFromA — never skipped. `fac` may be either factorization kind
 * (LDLᵀ A/B, plan 2026-07-06) — reuse solves dispatch on it.
 * @see oracle/tpe_constraints_oracle.py (Frozen / build_frozen_saddle)
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 6)
 * @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (pinned decision 4)
 */
export interface FrozenSaddleOperator {
    a: Float64Array;
    n: number;
    C: number[][];
    fac: SaddleFactorization;
}

/**
 * Solves K(γ₀)·[x; λ] = [rhsTop; rhsBottom] against a {@link FrozenSaddleOperator}
 * — forward/back substitution only, no assembly, no factorization ('factor'
 * never fires here; that is the point of the reuse). The relative residual is
 * ALWAYS computed against the frozen a/C via the same structured matvec as
 * solveSaddleFromA (spec §E prop 8 self-certification survives the reuse).
 * `rhsBottom` defaults to zeros(k); the frozen projection passes −Φ(γ^q) there.
 * @see oracle/tpe_constraints_oracle.py (solve_saddle_frozen)
 * @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 6)
 */
export function solveSaddleFrozen(
    op: FrozenSaddleOperator,
    rhsTop: number[],
    rhsBottom?: number[],
): { x: number[]; lambda: number[]; residual: number } {
    const { a, n, C, fac } = op;
    const m = 3 * n;
    const k = C.length;
    if (rhsTop.length !== m) {
        throw new Error(
            `solveSaddleFrozen: rhsTop length ${rhsTop.length} does not match 3n = ${m}`,
        );
    }
    const bottom = rhsBottom ?? new Array<number>(k).fill(0);
    if (bottom.length !== k) {
        throw new Error(
            `solveSaddleFrozen: rhsBottom length ${bottom.length} does not match C rows ${k}`,
        );
    }
    const rhs = [...rhsTop, ...bottom];
    const z = solveFactored(fac, rhs);
    const residual = structuredSaddleResidual(a, n, C, z, rhs);
    return { x: z.slice(0, m), lambda: z.slice(m), residual };
}
