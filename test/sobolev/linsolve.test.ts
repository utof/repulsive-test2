import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { buildSaddleMatrix, luSolve, solveSaddle } from '../../src/core/sobolev/linsolve';

// All 5 oracle fixture/golden pairs (Stage-1 Sobolev oracle harness).
// @see oracle/README.md
const FIXTURE_NAMES = ['crossing', 'junction-y', 'helix', 'linked-rings', 'knot'] as const;

interface Golden {
    A3: number[][];
    C_barycenter: number[][];
    dE_flat: number[];
    g_tilde_flat: number[];
    lambda: number[];
    saddle_relative_residual: number;
}

// Load at runtime (avoids needing resolveJsonModule in tsconfig; test/** is typechecked),
// mirroring test/sobolev/constraints.test.ts.
function loadGolden(name: string): Golden {
    return JSON.parse(
        readFileSync(new URL(`../../oracle/golden/${name}.json`, import.meta.url), 'utf8'),
    ) as Golden;
}

function euclideanNorm(a: number[]): number {
    let sumSq = 0;
    for (const x of a) sumSq += x * x;
    return Math.sqrt(sumSq);
}

function euclideanDiff(a: number[], b: number[]): number {
    let sumSq = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        sumSq += d * d;
    }
    return Math.sqrt(sumSq);
}

function maxAbsDiff(a: number[], b: number[]): number {
    let m = 0;
    for (let i = 0; i < a.length; i++) {
        m = Math.max(m, Math.abs(a[i] - b[i]));
    }
    return m;
}

function matVec(M: number[][], v: number[]): number[] {
    return M.map((row) => row.reduce((s, x, j) => s + x * v[j], 0));
}

// Deterministic 6×6 diagonally-dominant synthetic system with a known exact
// solution: rhs is built as K·x_known, so luSolve must recover x_known.
function syntheticSystem(): { K: number[][]; xKnown: number[]; rhs: number[] } {
    const n = 6;
    const K: number[][] = Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) => (i === j ? 10 + i : Math.sin(1 + 0.7 * i + 1.3 * j))),
    );
    const xKnown = Array.from({ length: n }, (_, k) => Math.cos(0.2 + 0.5 * k));
    return { K, xKnown, rhs: matVec(K, xKnown) };
}

test('luSolve: synthetic 6×6 exactness (‖x − x_known‖∞ ≤ 1e-12, residual ≤ 1e-14)', () => {
    const { K, xKnown, rhs } = syntheticSystem();
    const kCopy = K.map((row) => row.slice());
    const rhsCopy = rhs.slice();

    const x = luSolve(K, rhs);

    const errInf = maxAbsDiff(x, xKnown);
    const residual = euclideanDiff(matVec(K, x), rhs) / Math.max(1, euclideanNorm(rhs));
    console.log(
        `[linsolve] synthetic: ‖x − x_known‖∞ = ${errInf.toExponential(3)}, residual = ${residual.toExponential(3)}`,
    );
    expect(errInf).toBeLessThanOrEqual(1e-12);
    expect(residual).toBeLessThanOrEqual(1e-14);

    // Contract check: luSolve never mutates its inputs (it factors a copy).
    expect(K).toEqual(kCopy);
    expect(rhs).toEqual(rhsCopy);
});

test('luSolve: zero leading pivot forces a row swap and still solves', () => {
    const { K, xKnown } = syntheticSystem();
    K[0][0] = 0; // pivot column 0's max abs entry is now off-diagonal → row swap required
    const rhs = matVec(K, xKnown);

    const x = luSolve(K, rhs);

    const errInf = maxAbsDiff(x, xKnown);
    const residual = euclideanDiff(matVec(K, x), rhs) / Math.max(1, euclideanNorm(rhs));
    console.log(
        `[linsolve] pivoting: ‖x − x_known‖∞ = ${errInf.toExponential(3)}, residual = ${residual.toExponential(3)}`,
    );
    expect(errInf).toBeLessThanOrEqual(1e-12);
    expect(residual).toBeLessThanOrEqual(1e-14);
});

test('luSolve: genuinely singular matrix (two equal rows) throws mentioning "singular"', () => {
    const { K, rhs } = syntheticSystem();
    K[3] = K[2].slice(); // rank-deficient: rows 2 and 3 identical
    expect(() => luSolve(K, rhs)).toThrow(/singular/);
});

test('buildSaddleMatrix: symmetric layout [[A3, Cᵀ], [C, 0]] for any k', () => {
    const A3 = [
        [2, 1],
        [1, 3],
    ];
    const C = [[4, 5]]; // k = 1 — not the barycenter k=3, checks the any-k contract
    const K = buildSaddleMatrix(A3, C);
    expect(K).toEqual([
        [2, 1, 4],
        [1, 3, 5],
        [4, 5, 0],
    ]);
});

for (const name of FIXTURE_NAMES) {
    const golden = loadGolden(name);

    // Golden saddle systems — cross-SOLVER gate vs the independent Python
    // oracle (scipy la.solve with assume_a="sym", i.e. Bunch–Kaufman; we use
    // LU with partial pivoting, so agreement is bounded by conditioning).
    // The residual is the primary, self-certifying gate.
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §E (prop 8)
    // @see oracle/tpe_stage1_oracle.py (solve_saddle)
    test(`solveSaddle: ${name} — residual ≤ 1e-10, matches oracle g̃/λ to 1e-9, C·x ≈ 0`, () => {
        const { A3, C_barycenter: C, dE_flat, g_tilde_flat, lambda: goldenLambda } = golden;
        const { x, lambda, residual } = solveSaddle(A3, C, dE_flat);

        const xRelDiff = euclideanDiff(x, g_tilde_flat) / euclideanNorm(g_tilde_flat);
        const lambdaRelDiff =
            euclideanDiff(lambda, goldenLambda) / Math.max(1, euclideanNorm(goldenLambda));
        const constraintRel = euclideanNorm(matVec(C, x)) / Math.max(1, euclideanNorm(x));
        console.log(
            `[linsolve] ${name} (size ${A3.length + C.length}): residual = ${residual.toExponential(3)} ` +
                `(oracle: ${golden.saddle_relative_residual.toExponential(3)}), ` +
                `g̃ rel diff = ${xRelDiff.toExponential(3)}, λ rel diff = ${lambdaRelDiff.toExponential(3)}, ` +
                `‖C·x‖/max(1,‖x‖) = ${constraintRel.toExponential(3)}`,
        );

        // Self-certifying residual gate — spec §E prop 8.
        expect(residual).toBeLessThanOrEqual(1e-10);
        // Cross-solver agreement gates (LU here vs Bunch–Kaufman in scipy).
        expect(xRelDiff).toBeLessThanOrEqual(1e-9);
        expect(lambdaRelDiff).toBeLessThanOrEqual(1e-9);
        // Constraint satisfied: x lies in the null space of C — spec §E prop 8.
        expect(constraintRel).toBeLessThanOrEqual(1e-10);
    });
}
