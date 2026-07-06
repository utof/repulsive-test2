import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { sobolevStepSet } from '../../src/core/optimizer';
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
    type LdltFactorization,
    ldltFactor,
    ldltSolveFactored,
    luSolve,
    solveSaddle,
    solveSaddleFromA,
    solveSaddleFrozen,
} from '../../src/core/sobolev/linsolve';
import { calculateDisjointPairs, gradientAnalytical } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

// Bunch–Kaufman LDLᵀ (dsytf2-style) verification ladder, steps (a) and (b):
// unit reconstruction + solve-vs-LU cross-checks on deterministic seeded
// systems, then step-outcome equivalence LDLᵀ vs LU on every committed
// fixture. The number[][] reference path (luSolve/solveSaddle) stays LU by
// design and serves as the cross-check oracle here.
// @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (pinned decisions 1–5)
// @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (step 5.6 kill gates)

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

// Deterministic seeded PRNG (mulberry32) — TEST-ONLY randomness; runtime
// paths stay Math.random-free per the repo's determinism convention.
// @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (verification ladder a)
function mulberry32(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// Random symmetric INDEFINITE matrix: alternating-sign diagonal guarantees
// mixed eigenvalue signs (Gershgorin-ish), off-diagonal dense symmetric.
function randomSymmetricIndefinite(n: number, seed: number): number[][] {
    const rnd = mulberry32(seed);
    const A: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
            const v = 2 * (2 * rnd() - 1);
            A[i][j] = v;
            A[j][i] = v;
        }
        A[i][i] = (i % 2 === 0 ? 1 : -1) * (3 + 2 * rnd());
    }
    return A;
}

// Zero diagonal everywhere — every pivot decision is driven off-diagonal,
// forcing the 2×2 machinery hard (the saddle system's trailing block shape).
function zeroDiagSymmetric(n: number, seed: number): number[][] {
    const A = randomSymmetricIndefinite(n, seed);
    for (let i = 0; i < n; i++) A[i][i] = 0;
    return A;
}

function flattenSquare(M: number[][]): Float64Array {
    const n = M.length;
    const out = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) out[i * n + j] = M[i][j];
    }
    return out;
}

function matMul(A: number[][], B: number[][]): number[][] {
    const n = A.length;
    const out: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (let i = 0; i < n; i++) {
        for (let k = 0; k < n; k++) {
            const a = A[i][k];
            if (a === 0) continue;
            for (let j = 0; j < n; j++) out[i][j] += a * B[k][j];
        }
    }
    return out;
}

function transpose(A: number[][]): number[][] {
    return A.map((_, i) => A.map((row) => row[i]));
}

function identityMatrix(n: number): number[][] {
    return Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
    );
}

function matVec(M: number[][], v: number[]): number[] {
    return M.map((row) => row.reduce((s, x, j) => s + x * v[j], 0));
}

function frobenius(M: number[][]): number {
    let s = 0;
    for (const row of M) for (const x of row) s += x * x;
    return Math.sqrt(s);
}

function frobeniusDiff(A: number[][], B: number[][]): number {
    let s = 0;
    for (let i = 0; i < A.length; i++) {
        for (let j = 0; j < A.length; j++) {
            const d = A[i][j] - B[i][j];
            s += d * d;
        }
    }
    return Math.sqrt(s);
}

function euclideanNorm(a: number[]): number {
    let s = 0;
    for (const x of a) s += x * x;
    return Math.sqrt(s);
}

function euclideanDiff(a: number[], b: number[]): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        s += d * d;
    }
    return Math.sqrt(s);
}

// Reconstructs A from the packed Bunch–Kaufman factors via LAPACK dsytrf's
// documented product form A = P(1)·L(1)···P(s)·L(s)·D·L(s)ᵀ·P(s)ᵀ···L(1)ᵀ·P(1)ᵀ
// (interchanges are interleaved — columns < k of L are NOT re-permuted, which
// is why a plain P·A·Pᵀ = L·D·Lᵀ comparison would be wrong).
// @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (pinned decision 1)
function reconstructFromLdlt(fac: LdltFactorization): number[][] {
    const { m, ipiv, n } = fac;
    const steps: { k: number; kstep: number; kp: number }[] = [];
    let k = 0;
    while (k < n) {
        if (ipiv[k] > 0) {
            steps.push({ k, kstep: 1, kp: ipiv[k] - 1 });
            k += 1;
        } else {
            steps.push({ k, kstep: 2, kp: -ipiv[k] - 1 });
            k += 2;
        }
    }
    // D: block diagonal from the packed diagonal (+ subdiagonal at 2×2 blocks).
    let M: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (const s of steps) {
        M[s.k][s.k] = m[s.k * n + s.k];
        if (s.kstep === 2) {
            const off = m[(s.k + 1) * n + s.k];
            M[s.k + 1][s.k] = off;
            M[s.k][s.k + 1] = off;
            M[s.k + 1][s.k + 1] = m[(s.k + 1) * n + s.k + 1];
        }
    }
    // Wrap D with L(s)·…·L(1) and the interchanges, last step innermost.
    for (const s of [...steps].reverse()) {
        const L = identityMatrix(n);
        const lo = s.k + s.kstep; // multipliers start below the pivot block
        for (let i = lo; i < n; i++) {
            L[i][s.k] = m[i * n + s.k];
            if (s.kstep === 2) L[i][s.k + 1] = m[i * n + s.k + 1];
        }
        M = matMul(matMul(L, M), transpose(L));
        const kk = s.k + s.kstep - 1;
        if (s.kp !== kk) {
            const tmpRow = M[kk];
            M[kk] = M[s.kp];
            M[s.kp] = tmpRow;
            for (const row of M) {
                const t = row[kk];
                row[kk] = row[s.kp];
                row[s.kp] = t;
            }
        }
    }
    return M;
}

// ── (a) unit: reconstruction + solve vs the LU reference ────────────────────

for (const { n, seed, kind } of [
    { n: 7, seed: 1, kind: 'indefinite' },
    { n: 12, seed: 42, kind: 'indefinite' },
    { n: 25, seed: 2026, kind: 'indefinite' },
    { n: 12, seed: 7, kind: 'zero-diag' },
    { n: 24, seed: 99, kind: 'zero-diag' },
] as const) {
    test(`ldltFactor: ${kind} ${n}×${n} (seed ${seed}) — reconstruction rel ≤ 1e-12, solve vs luSolve rel ≤ 1e-9, residual ≤ 1e-12`, () => {
        const A =
            kind === 'zero-diag' ? zeroDiagSymmetric(n, seed) : randomSymmetricIndefinite(n, seed);
        const rnd = mulberry32(seed ^ 0x5bd1e995);
        const b = Array.from({ length: n }, () => 2 * rnd() - 1);

        const fac = ldltFactor(flattenSquare(A), n);
        const rebuilt = reconstructFromLdlt(fac);
        const recRel = frobeniusDiff(rebuilt, A) / Math.max(1, frobenius(A));
        expect(recRel).toBeLessThanOrEqual(1e-12);

        const x = ldltSolveFactored(fac, b);
        const xRef = luSolve(A, b);
        const solRel = euclideanDiff(x, xRef) / Math.max(1, euclideanNorm(xRef));
        expect(solRel).toBeLessThanOrEqual(1e-9);

        const resid = euclideanDiff(matVec(A, x), b) / Math.max(1, euclideanNorm(b));
        expect(resid).toBeLessThanOrEqual(1e-12);
    });
}

test('ldltFactor: 2×2 exchange matrix [[0,1],[1,0]] takes a 2×2 pivot (ipiv = [-2,-2]) and solves exactly', () => {
    const fac = ldltFactor(
        flattenSquare([
            [0, 1],
            [1, 0],
        ]),
        2,
    );
    expect(Array.from(fac.ipiv)).toEqual([-2, -2]);
    // Exchange matrix swaps the rhs — exact in floating point.
    expect(ldltSolveFactored(fac, [3, -5])).toEqual([-5, 3]);
});

test('ldltFactor: zero matrix throws mentioning "singular" (never reaches the solve)', () => {
    expect(() => ldltFactor(new Float64Array(9), 3)).toThrow(/singular/);
});

test('ldltFactor: non-finite entry throws "treating as singular", matching luFactor contract', () => {
    const A = randomSymmetricIndefinite(6, 3);
    A[2][4] = Number.NaN;
    A[4][2] = Number.NaN;
    expect(() => ldltFactor(flattenSquare(A), 6)).toThrow(/treating as singular/);
});

test('ldltSolveFactored: non-finite rhs throws "treating as singular", matching luSolveFactored contract', () => {
    const A = randomSymmetricIndefinite(6, 4);
    const fac = ldltFactor(flattenSquare(A), 6);
    const bad = [1, 2, Number.POSITIVE_INFINITY, 4, 5, 6];
    expect(() => ldltSolveFactored(fac, bad)).toThrow(/treating as singular/);
});

// ── (a) real saddle systems: factorMode 'ldlt' vs the slow LU reference ─────

const crossing = loadFixture('crossing');
const crossingPairs = calculateDisjointPairs(crossing.edges);
const crossingSet: ConstraintSet = [
    barycenterBlock(barycenterTarget(crossing.vertices, crossing.edges)),
    totalLengthBlock(totalLength(crossing.vertices, crossing.edges)),
];

test("solveSaddleFromA factorMode 'ldlt' vs solveSaddle (LU): crossing gradient system — x/λ rel ≤ 1e-9, residuals ≤ 1e-10", () => {
    const { vertices, edges, alpha, beta, epsilon } = crossing;
    const n = vertices.length;
    const A = assembleA(vertices, edges, crossingPairs, alpha, beta, epsilon);
    const aFlat = assembleAFlat(vertices, edges, crossingPairs, alpha, beta, epsilon);
    const { C } = evaluateConstraintSet(crossingSet, vertices, edges);
    const dEFlat = flatten(
        gradientAnalytical(vertices, edges, crossingPairs, alpha, beta, epsilon),
    );

    const slow = solveSaddle(expandBlockDiag(A), C, dEFlat);
    const fast = solveSaddleFromA(aFlat, n, C, dEFlat, undefined, 'ldlt');

    const xRel = euclideanDiff(fast.x, slow.x) / Math.max(1, euclideanNorm(slow.x));
    const lambdaRel =
        euclideanDiff(fast.lambda, slow.lambda) / Math.max(1, euclideanNorm(slow.lambda));
    console.log(
        `[ldlt] crossing gradient: x rel = ${xRel.toExponential(3)}, λ rel = ${lambdaRel.toExponential(3)}, ` +
            `residual ldlt = ${fast.residual.toExponential(3)}, lu = ${slow.residual.toExponential(3)}`,
    );
    expect(xRel).toBeLessThanOrEqual(1e-9);
    expect(lambdaRel).toBeLessThanOrEqual(1e-9);
    expect(fast.residual).toBeLessThanOrEqual(1e-10);
    expect(slow.residual).toBeLessThanOrEqual(1e-10);
    expect(fast.fac).toHaveProperty('kind', 'ldlt');
    expect(fast.fac.n).toBe(3 * n + C.length);
});

test("solveSaddleFromA factorMode 'ldlt': crossing projection system (rhsBottom = −Φ) vs solveSaddle — rel ≤ 1e-9, residuals ≤ 1e-10", () => {
    const { vertices, edges, alpha, beta, epsilon } = crossing;
    const n = vertices.length;
    const perturbed: Vec3[] = vertices.map((p) => [1.02 * p[0], 1.02 * p[1] + 0.01, 1.02 * p[2]]);
    const A = assembleA(perturbed, edges, crossingPairs, alpha, beta, epsilon);
    const aFlat = assembleAFlat(perturbed, edges, crossingPairs, alpha, beta, epsilon);
    const { phi, C } = evaluateConstraintSet(crossingSet, perturbed, edges);
    const negPhi = phi.map((v) => -v);
    const zeroTop = new Array<number>(3 * n).fill(0);

    const slow = solveSaddle(expandBlockDiag(A), C, zeroTop, negPhi);
    const fast = solveSaddleFromA(aFlat, n, C, zeroTop, negPhi, 'ldlt');

    const xRel = euclideanDiff(fast.x, slow.x) / Math.max(1, euclideanNorm(slow.x));
    const lambdaRel =
        euclideanDiff(fast.lambda, slow.lambda) / Math.max(1, euclideanNorm(slow.lambda));
    expect(xRel).toBeLessThanOrEqual(1e-9);
    expect(lambdaRel).toBeLessThanOrEqual(1e-9);
    expect(fast.residual).toBeLessThanOrEqual(1e-10);
});

test("solveSaddleFrozen: reuses an 'ldlt' factorization for a fresh rhs — self-certifying residual ≤ 1e-10 survives the reuse", () => {
    const { vertices, edges, alpha, beta, epsilon } = crossing;
    const n = vertices.length;
    const aFlat = assembleAFlat(vertices, edges, crossingPairs, alpha, beta, epsilon);
    const { C } = evaluateConstraintSet(crossingSet, vertices, edges);
    const dEFlat = flatten(
        gradientAnalytical(vertices, edges, crossingPairs, alpha, beta, epsilon),
    );

    const first = solveSaddleFromA(aFlat, n, C, dEFlat, undefined, 'ldlt');
    const op = { a: aFlat, n, C, fac: first.fac };
    // Fresh projection-shaped rhs against the SAME frozen factorization.
    const negPhi = C.map((_, r) => 0.01 * (r + 1));
    const reuse = solveSaddleFrozen(op, new Array<number>(3 * n).fill(0), negPhi);
    expect(reuse.residual).toBeLessThanOrEqual(1e-10);
});

test("solveSaddleFromA factorMode 'ldlt': k = 0 (empty constraint set) degenerates to Ā·x = rhs (spec §9a) — certified residual", () => {
    const { vertices, edges, alpha, beta, epsilon } = crossing;
    const n = vertices.length;
    const aFlat = assembleAFlat(vertices, edges, crossingPairs, alpha, beta, epsilon);
    const dEFlat = flatten(
        gradientAnalytical(vertices, edges, crossingPairs, alpha, beta, epsilon),
    );

    // NO solution comparison against the LU reference here: unconstrained Ā
    // is numerically singular (constant fields are its null space — the very
    // reason the barycenter constraint exists), so two DIFFERENT
    // factorizations legitimately return solutions differing by a null-space
    // component while both certify residual ~1e-16 (measured 2026-07-06:
    // xRel ≈ 0.67 with both residuals ≤ 2e-16). The self-certifying residual
    // is the correctness statement; linsolveFlat.test.ts's exact k=0 LU-vs-LU
    // comparison only holds because both paths run identical ops.
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("Do not 'fix' singularity…")
    // @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (verification ladder a)
    const fast = solveSaddleFromA(aFlat, n, [], dEFlat, undefined, 'ldlt');
    expect(fast.lambda).toEqual([]);
    expect(fast.residual).toBeLessThanOrEqual(1e-10);
});

// ── (b) property: step outcomes LDLᵀ ≡ LU on every committed fixture ─────────

const FIXTURES = ['crossing', 'helix', 'junction-y', 'knot', 'linked-rings'] as const;

for (const name of FIXTURES) {
    test(`sobolevStepSet factorMode 'ldlt' vs default LU: ${name} — same verdicts, vertices rel ≤ 1e-9, residuals ≤ 1e-10`, () => {
        const { vertices, edges, alpha, beta, epsilon } = loadFixture(name);
        const disjointPairs = calculateDisjointPairs(edges);
        const set: ConstraintSet = [
            barycenterBlock(barycenterTarget(vertices, edges)),
            totalLengthBlock(totalLength(vertices, edges)),
        ];
        const base = { mode: 'analytical' as const, alpha, beta, epsilon };

        const lu = sobolevStepSet(vertices, edges, disjointPairs, set, base);
        const ldlt = sobolevStepSet(vertices, edges, disjointPairs, set, {
            ...base,
            factorMode: 'ldlt',
        });

        expect(ldlt.accepted).toBe(lu.accepted);
        expect(ldlt.converged).toBe(lu.converged);
        expect(lu.stats.residual).toBeLessThanOrEqual(1e-10);
        expect(ldlt.stats.residual).toBeLessThanOrEqual(1e-10);
        const vRel =
            euclideanDiff(flatten(ldlt.vertices), flatten(lu.vertices)) /
            Math.max(1, euclideanNorm(flatten(lu.vertices)));
        console.log(
            `[ldlt] ${name}: accepted=${lu.accepted} vRel=${vRel.toExponential(3)} ` +
                `resid lu=${lu.stats.residual.toExponential(3)} ldlt=${ldlt.stats.residual.toExponential(3)}`,
        );
        expect(vRel).toBeLessThanOrEqual(1e-9);
    });
}

test("sobolevStepSet: projectionMode 'frozen' + factorMode 'ldlt' (frozen op carries an LDLᵀ factorization) matches frozen+LU on crossing", () => {
    const { vertices, edges, alpha, beta, epsilon } = crossing;
    const disjointPairs = calculateDisjointPairs(edges);
    const set: ConstraintSet = [
        barycenterBlock(barycenterTarget(vertices, edges)),
        totalLengthBlock(totalLength(vertices, edges)),
    ];
    const base = {
        mode: 'analytical' as const,
        alpha,
        beta,
        epsilon,
        projectionMode: 'frozen' as const,
    };

    const lu = sobolevStepSet(vertices, edges, disjointPairs, set, base);
    const ldlt = sobolevStepSet(vertices, edges, disjointPairs, set, {
        ...base,
        factorMode: 'ldlt',
    });

    expect(ldlt.accepted).toBe(lu.accepted);
    expect(ldlt.stats.residual).toBeLessThanOrEqual(1e-10);
    const vRel =
        euclideanDiff(flatten(ldlt.vertices), flatten(lu.vertices)) /
        Math.max(1, euclideanNorm(flatten(lu.vertices)));
    expect(vRel).toBeLessThanOrEqual(1e-9);
});

test("sobolevStepSet factorMode 'ldlt': singular saddle (isolated vertex) → 'singular_system' rejection, never a throw", () => {
    const { vertices, edges, alpha, beta, epsilon } = crossing;
    // Isolated vertex ⇒ zero Ā rows / C columns ⇒ exactly singular saddle —
    // the LDLᵀ breakdown must fold into the same reject contract as LU.
    // @see docs/superpowers/plans/2026-07-06-ldlt-factor.md (pinned decision 3)
    const withIsolated: Vec3[] = [...vertices.map((v) => [v[0], v[1], v[2]] as Vec3), [5, 5, 5]];
    const disjointPairs = calculateDisjointPairs(edges);
    const set: ConstraintSet = [
        barycenterBlock(barycenterTarget(withIsolated, edges)),
        totalLengthBlock(totalLength(withIsolated, edges)),
    ];

    const r = sobolevStepSet(withIsolated, edges, disjointPairs, set, {
        mode: 'analytical',
        alpha,
        beta,
        epsilon,
        factorMode: 'ldlt',
    });
    expect(r.accepted).toBe(false);
    expect(r.stats.reason).toBe('singular_system');
    expect(r.vertices).toEqual(withIsolated);
});
