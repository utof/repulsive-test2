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
): { x: number[]; lambda: number[]; residual: number; fac: LuFactorization } {
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
    const fac = timed('factor', () => luFactor(K, size));
    const z = luSolveFactored(fac, rhs);

    // Self-certifying residual — computed on EVERY solve, never skipped.
    // Structured matvec against the original a/C (the factorization destroyed K).
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B / §E (prop 8: saddle residual)
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
    const residual = Math.sqrt(resNormSq) / Math.max(1, Math.sqrt(rhsNormSq));

    return { x: z.slice(0, m), lambda: z.slice(m), residual, fac };
}
