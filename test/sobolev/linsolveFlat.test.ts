import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import {
    barycenterBlock,
    type ConstraintSet,
    evaluateConstraintSet,
    totalLength,
    totalLengthBlock,
} from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import { assembleA, assembleAFlat } from '../../src/core/sobolev/innerProduct';
import { expandBlockDiag, flatten } from '../../src/core/sobolev/layout';
import {
    luFactor,
    luSolve,
    luSolveFactored,
    solveSaddle,
    solveSaddleFromA,
} from '../../src/core/sobolev/linsolve';
import { calculateDisjointPairs, gradientAnalytical } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

// Typed-array dense core (plan Task 5): the flat fast path (assembleAFlat +
// luFactor/luSolveFactored + solveSaddleFromA) is cross-checked against the
// retained slow reference implementations (assembleA / luSolve / solveSaddle),
// which stay golden-tested. The flat path performs the SAME op sequence on the
// same values, so agreement is exact where the plan demands it.
// @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 5, step 5.1)
// @see local_files/2026-07-03-next-steps-briefing.md §5A item 2

interface Fixture {
    name: string;
    vertices: Vec3[];
    edges: Edge[];
    alpha: number;
    beta: number;
    epsilon: number;
}

function loadFixture(name: string): Fixture {
    return JSON.parse(
        readFileSync(new URL(`../../oracle/fixtures/${name}.json`, import.meta.url), 'utf8'),
    ) as Fixture;
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

function matVec(M: number[][], v: number[]): number[] {
    return M.map((row) => row.reduce((s, x, j) => s + x * v[j], 0));
}

function flattenSquare(M: number[][]): Float64Array {
    const n = M.length;
    const out = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) out[i * n + j] = M[i][j];
    }
    return out;
}

function unflattenSquare(flat: Float64Array, n: number): number[][] {
    const out: number[][] = new Array(n);
    for (let i = 0; i < n; i++) {
        const row = new Array<number>(n);
        for (let j = 0; j < n; j++) row[j] = flat[i * n + j];
        out[i] = row;
    }
    return out;
}

// Deterministic well-conditioned 12×12 system (diagonally dominant), same
// recipe as linsolve.test.ts's syntheticSystem but at n = 12 per plan 5.1.
function synthetic12(): { K: number[][]; rhs: number[] } {
    const n = 12;
    const K: number[][] = Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) => (i === j ? 10 + i : Math.sin(1 + 0.7 * i + 1.3 * j))),
    );
    const xKnown = Array.from({ length: n }, (_, k) => Math.cos(0.2 + 0.5 * k));
    return { K, rhs: matVec(K, xKnown) };
}

test('luFactor + luSolveFactored ≡ luSolve exactly on a deterministic 12×12 system', () => {
    const { K, rhs } = synthetic12();
    const reference = luSolve(K, rhs);

    const buf = flattenSquare(K); // luFactor consumes the buffer (in-place)
    const fac = luFactor(buf, 12);
    const x = luSolveFactored(fac, rhs);

    // Identical algorithm and op order on the same values → bit-identical.
    // @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 5, step 5.1)
    expect(x).toEqual(reference);
});

test('luFactor: zero pivot column throws mentioning "singular", matching luSolve', () => {
    const { K } = synthetic12();
    for (let r = 0; r < K.length; r++) K[r][2] = 0; // column 2 identically zero
    const { rhs } = synthetic12();

    expect(() => luSolve(K, rhs)).toThrow(/singular/);
    expect(() => luFactor(flattenSquare(K), 12)).toThrow(/singular/);
});

test('luFactor: non-finite matrix entry throws "treating as singular", matching luSolve', () => {
    const { K, rhs } = synthetic12();
    K[3][7] = Number.NaN;

    expect(() => luSolve(K, rhs)).toThrow(/treating as singular/);
    expect(() => luFactor(flattenSquare(K), 12)).toThrow(/treating as singular/);
});

test('luSolveFactored: non-finite rhs entry throws "treating as singular", matching luSolve', () => {
    const { K, rhs } = synthetic12();
    const badRhs = rhs.slice();
    badRhs[5] = Number.POSITIVE_INFINITY;

    expect(() => luSolve(K, badRhs)).toThrow(/treating as singular/);
    const fac = luFactor(flattenSquare(K), 12);
    expect(() => luSolveFactored(fac, badRhs)).toThrow(/treating as singular/);
});

// ── solveSaddleFromA vs solveSaddle on the crossing fixture's REAL system ────

const crossing = loadFixture('crossing');
const crossingPairs = calculateDisjointPairs(crossing.edges);
const crossingSet: ConstraintSet = [
    barycenterBlock(barycenterTarget(crossing.vertices, crossing.edges)),
    totalLengthBlock(totalLength(crossing.vertices, crossing.edges)),
];

test('solveSaddleFromA vs solveSaddle: crossing gradient system — x/λ rel ≤ 1e-12, both residuals ≤ 1e-10', () => {
    const { vertices, edges, alpha, beta, epsilon } = crossing;
    const n = vertices.length;
    const A = assembleA(vertices, edges, crossingPairs, alpha, beta, epsilon);
    const aFlat = assembleAFlat(vertices, edges, crossingPairs, alpha, beta, epsilon);
    const { C } = evaluateConstraintSet(crossingSet, vertices, edges);
    const dEFlat = flatten(
        gradientAnalytical(vertices, edges, crossingPairs, alpha, beta, epsilon),
    );

    const slow = solveSaddle(expandBlockDiag(A), C, dEFlat);
    // factorMode 'lu' PINNED: this test's claim is "flat LU fast path ≡ slow
    // LU reference, same op sequence" — it must keep testing the LU leg after
    // the 2026-07-06 default flip to 'ldlt'. LDLᵀ-vs-LU lives in ldlt.test.ts.
    // @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (pinned decision 2)
    const fast = solveSaddleFromA(aFlat, n, C, dEFlat, undefined, 'lu');

    const xRel = euclideanDiff(fast.x, slow.x) / Math.max(1, euclideanNorm(slow.x));
    const lambdaRel =
        euclideanDiff(fast.lambda, slow.lambda) / Math.max(1, euclideanNorm(slow.lambda));
    console.log(
        `[linsolveFlat] crossing gradient: x rel = ${xRel.toExponential(3)}, λ rel = ${lambdaRel.toExponential(3)}, ` +
            `residual fast = ${fast.residual.toExponential(3)}, slow = ${slow.residual.toExponential(3)}`,
    );
    expect(xRel).toBeLessThanOrEqual(1e-12);
    expect(lambdaRel).toBeLessThanOrEqual(1e-12);
    expect(fast.residual).toBeLessThanOrEqual(1e-10);
    expect(slow.residual).toBeLessThanOrEqual(1e-10);
    // Task 6 consumes the returned factorization — it must describe the full
    // (3n+k)×(3n+k) saddle system.
    expect(fast.fac.n).toBe(3 * n + C.length);
});

test('solveSaddleFromA vs solveSaddle: crossing projection system (rhsBottom = −Φ) — x/λ rel ≤ 1e-12, residuals ≤ 1e-10', () => {
    const { vertices, edges, alpha, beta, epsilon } = crossing;
    const n = vertices.length;
    // A real projection iterate: perturb the curve off the frozen targets so
    // Φ ≠ 0, then solve [[Ā, Cᵀ], [C, 0]]·[x; μ] = [0; −Φ] at the perturbed
    // geometry — the exact system projectOntoConstraintSet solves.
    const perturbed: Vec3[] = vertices.map((p) => [1.02 * p[0], 1.02 * p[1] + 0.01, 1.02 * p[2]]);
    const A = assembleA(perturbed, edges, crossingPairs, alpha, beta, epsilon);
    const aFlat = assembleAFlat(perturbed, edges, crossingPairs, alpha, beta, epsilon);
    const { phi, C } = evaluateConstraintSet(crossingSet, perturbed, edges);
    const negPhi = phi.map((v) => -v);
    const zeroTop = new Array<number>(3 * n).fill(0);

    const slow = solveSaddle(expandBlockDiag(A), C, zeroTop, negPhi);
    // factorMode 'lu' PINNED — LU-vs-LU identity test (see the gradient-system
    // test above). @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (decision 2)
    const fast = solveSaddleFromA(aFlat, n, C, zeroTop, negPhi, 'lu');

    const xRel = euclideanDiff(fast.x, slow.x) / Math.max(1, euclideanNorm(slow.x));
    const lambdaRel =
        euclideanDiff(fast.lambda, slow.lambda) / Math.max(1, euclideanNorm(slow.lambda));
    console.log(
        `[linsolveFlat] crossing projection: x rel = ${xRel.toExponential(3)}, λ rel = ${lambdaRel.toExponential(3)}, ` +
            `residual fast = ${fast.residual.toExponential(3)}, slow = ${slow.residual.toExponential(3)}`,
    );
    expect(xRel).toBeLessThanOrEqual(1e-12);
    expect(lambdaRel).toBeLessThanOrEqual(1e-12);
    expect(fast.residual).toBeLessThanOrEqual(1e-10);
    expect(slow.residual).toBeLessThanOrEqual(1e-10);
});

test('solveSaddleFromA: k = 0 (empty constraint set) degenerates to Ā·x = rhs like solveSaddle (spec §9a)', () => {
    const { vertices, edges, alpha, beta, epsilon } = crossing;
    const n = vertices.length;
    const A = assembleA(vertices, edges, crossingPairs, alpha, beta, epsilon);
    const aFlat = assembleAFlat(vertices, edges, crossingPairs, alpha, beta, epsilon);
    const dEFlat = flatten(
        gradientAnalytical(vertices, edges, crossingPairs, alpha, beta, epsilon),
    );

    const slow = solveSaddle(expandBlockDiag(A), [], dEFlat);
    // factorMode 'lu' PINNED — and here it is LOAD-BEARING, not just intent:
    // unconstrained Ā is numerically singular (constant null space), so only
    // an op-identical factorization reproduces the same solution; LDLᵀ picks a
    // different null component (measured xRel ≈ 0.67 with both residuals
    // ≤ 2e-16 — see ldlt.test.ts's k=0 test).
    // @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (pinned decision 2)
    const fast = solveSaddleFromA(aFlat, n, [], dEFlat, undefined, 'lu');

    const xRel = euclideanDiff(fast.x, slow.x) / Math.max(1, euclideanNorm(slow.x));
    expect(xRel).toBeLessThanOrEqual(1e-12);
    expect(fast.lambda).toEqual([]);
    expect(fast.residual).toBeLessThanOrEqual(1e-10);
});

// ── assembleAFlat vs assembleA: entrywise exact on two fixtures ──────────────

for (const name of ['crossing', 'knot'] as const) {
    test(`assembleAFlat ≡ assembleA entrywise (toEqual) on ${name}`, () => {
        const { vertices, edges, alpha, beta, epsilon } = loadFixture(name);
        const disjointPairs = calculateDisjointPairs(edges);
        const n = vertices.length;

        const nested = assembleA(vertices, edges, disjointPairs, alpha, beta, epsilon);
        const flat = assembleAFlat(vertices, edges, disjointPairs, alpha, beta, epsilon);

        // Exact: the flat body performs the verbatim op sequence of the nested
        // one (same accumulation order, same per-matrix symmetrization), only
        // the storage layout differs. @see plan Task 5 step 5.1.
        expect(unflattenSquare(flat, n)).toEqual(nested);
    });
}
