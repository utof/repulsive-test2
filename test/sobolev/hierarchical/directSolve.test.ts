/**
 * The direct variant of the matrix-free route: exact assembly, Cholesky, and the per-geometry
 * choice between it and the tree. @see src/core/sobolev/hierarchical/directSolve.ts
 */
import { describe, expect, setDefaultTimeout, test } from 'bun:test';
import { barnesHutGradient, prepareBarnesHut } from '../../../src/core/barnesHut';
import {
    assembleScalarA,
    choleskyInPlace,
    DIRECT_MAX_N,
    directIsCheaper,
    makeDirectSolve,
} from '../../../src/core/sobolev/hierarchical/directSolve';
import {
    makeDirectSaddle,
    makeRingSaddle,
} from '../../../src/core/sobolev/hierarchical/hierarchicalSaddle';
import { assembleAFlat } from '../../../src/core/sobolev/innerProduct';
import { calculateDisjointPairs, gradientAnalytical } from '../../../src/core/tangentPointEnergy';
import { type Edge, testConfigs, type Vec3 } from '../../../src/core/testConfigs';

setDefaultTimeout(120_000);

function lcg(seed: number): () => number {
    let s = seed;
    return () => {
        s = (1664525 * s + 1013904223) % 4294967296;
        return s / 4294967296;
    };
}
/** A closed polygon through random points of a cube: a tight tangle, the direct route's case. */
function tangle(n: number, seed: number): { V: Vec3[]; E: Edge[] } {
    const r = lcg(seed);
    const V: Vec3[] = Array.from({ length: n }, () => [r(), r(), r()]);
    return { V, E: V.map((_, i) => [i, (i + 1) % n] as Edge) };
}
const largeKnot = testConfigs.find((t) => t.id === 'knot-large');
if (!largeKnot) throw new Error('the knot-large preset is missing');
const relDiff = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
    let num = 0;
    let den = 0;
    for (let i = 0; i < a.length; i++) {
        num += (a[i] - b[i]) ** 2;
        den += b[i] ** 2;
    }
    return Math.sqrt(num / den);
};

describe('assembleScalarA', () => {
    test('equals the reference assembly on a ring, to rounding, and is exactly symmetric', () => {
        const { V, E } = tangle(120, 1);
        const ref = assembleAFlat(V, E, calculateDisjointPairs(E), 3, 6, 1e-10);
        const fast = assembleScalarA(V, E, 3, 6, 1e-10);
        // MEASURED 1.2e-15 at n = 400.
        expect(relDiff(fast, ref)).toBeLessThan(1e-12);
        const n = V.length;
        for (let i = 0; i < n; i++)
            for (let j = 0; j < i; j++) expect(fast[i * n + j]).toBe(fast[j * n + i]);
    });

    test('equals the reference on a graph with junctions and with other exponents', () => {
        // A ring with three chords: vertices of degree 3, so "shares a vertex" is not "|I−J| ≤ 1".
        const { V, E } = tangle(40, 2);
        const edges: Edge[] = [...E, [0, 20], [5, 30], [10, 25]];
        const pairs = calculateDisjointPairs(edges);
        for (const [alpha, beta] of [
            [3, 6],
            [2, 4.5],
        ]) {
            const ref = assembleAFlat(V, edges, pairs, alpha, beta, 1e-10);
            expect(relDiff(assembleScalarA(V, edges, alpha, beta, 1e-10), ref)).toBeLessThan(1e-12);
        }
    });
});

describe('choleskyInPlace', () => {
    test('factors a positive definite matrix: L·Lᵀ reproduces it', () => {
        const n = 37; // not a multiple of 4: exercises the unrolled loop's tail
        const r = lcg(3);
        const B = Float64Array.from({ length: n * n }, () => r() - 0.5);
        const M = new Float64Array(n * n);
        for (let i = 0; i < n; i++)
            for (let j = 0; j < n; j++) {
                let s = i === j ? n : 0;
                for (let k = 0; k < n; k++) s += B[i * n + k] * B[j * n + k];
                M[i * n + j] = s;
            }
        const L = Float64Array.from(M);
        expect(choleskyInPlace(L, n)).toBe(true);
        let worst = 0;
        for (let i = 0; i < n; i++)
            for (let j = 0; j <= i; j++) {
                let s = 0;
                for (let k = 0; k <= j; k++) s += L[i * n + k] * L[j * n + k];
                worst = Math.max(worst, Math.abs(s - M[i * n + j]));
            }
        expect(worst).toBeLessThan(1e-12);
    });

    test('refuses a matrix that is not positive definite', () => {
        expect(choleskyInPlace(Float64Array.from([1, 2, 2, 1]), 2)).toBe(false);
        expect(choleskyInPlace(Float64Array.from([0, 0, 0, 1]), 2)).toBe(false);
        expect(choleskyInPlace(Float64Array.from([Number.NaN, 0, 0, 1]), 2)).toBe(false);
    });
});

describe('makeDirectSolve', () => {
    test('the preconditioner is the exact inverse of the shifted operator', () => {
        const { V, E } = tangle(150, 4);
        const d = makeDirectSolve(V, E, 3, 6, 1e-10);
        if (!d) throw new Error('the factorization failed');
        const n = V.length;
        const u = Float64Array.from({ length: 3 * n }, (_, i) => Math.sin(1.7 * i));
        const b = new Float64Array(3 * n);
        d.op.applyA(u, b);
        // A_σ u = A u + (σ/n)(Σu)·1, block by block — the shift `gmres.ts` applies.
        for (let c = 0; c < 3; c++) {
            let sum = 0;
            for (let i = 0; i < n; i++) sum += u[c * n + i];
            for (let i = 0; i < n; i++) b[c * n + i] += (d.op.sigma / n) * sum;
        }
        const x = new Float64Array(3 * n);
        d.precond.apply(b, x);
        // MEASURED 2.9e-13 at n = 1 998.
        expect(relDiff(x, u)).toBeLessThan(1e-10);
        // σ is the trace of the exact matrix over n, and A annihilates constants.
        const ones = new Float64Array(3 * n).fill(1);
        const a1 = new Float64Array(3 * n);
        d.op.applyA(ones, a1);
        expect(Math.max(...a1.map(Math.abs))).toBeLessThan(1e-9 * d.op.sigma * n);
        expect(d.precond.sigma).toBe(d.op.sigma);
    });

    test('a curve with two coincident vertices is refused, not factored into garbage', () => {
        const { V, E } = tangle(30, 5);
        V[7] = [...V[20]] as Vec3;
        const d = makeDirectSolve(V, E, 3, 6, 1e-10);
        // Either refusal is acceptable; what is not is a "factor" full of NaN.
        if (d) {
            const out = new Float64Array(3 * V.length);
            d.precond.apply(new Float64Array(3 * V.length).fill(1), out);
            expect(out.every(Number.isFinite)).toBe(true);
        } else expect(d).toBeNull();
    });
});

describe('directIsCheaper', () => {
    test('the threshold is the documented near-field fraction, and the cap is hard', () => {
        // 0.077 + 3.8e-5·n: 9.6% at 500, 15.3% at 2 000.
        expect(directIsCheaper(500, 0.095 * 500 * 500)).toBe(false);
        expect(directIsCheaper(500, 0.097 * 500 * 500)).toBe(true);
        expect(directIsCheaper(2000, 0.152 * 2000 * 2000)).toBe(false);
        expect(directIsCheaper(2000, 0.154 * 2000 * 2000)).toBe(true);
        // The measured rows the rule must get right (bench/results/2026-11-23-direct-vs-tree.jsonl):
        // smooth trefoils and the dataset knot along its run, near-field fraction → faster solver.
        expect(directIsCheaper(500, 0.204 * 500 * 500)).toBe(true); // 0.36 s direct, 0.60 s tree
        expect(directIsCheaper(1000, 0.091 * 1000 * 1000)).toBe(false); // 1.19 s / 0.73 s
        expect(directIsCheaper(1998, 0.278 * 1998 * 1998)).toBe(true); // 5.6 s / 11.8 s
        expect(directIsCheaper(1998, 0.088 * 1998 * 1998)).toBe(false); // 5.6 s / 4.5 s
        expect(directIsCheaper(DIRECT_MAX_N + 1, (DIRECT_MAX_N + 1) ** 2)).toBe(false);
    });
});

describe('makeRingSaddle', () => {
    test("'auto' takes the direct route on a tangle and the tree on a large smooth ring", () => {
        const t = tangle(300, 6);
        const a = makeRingSaddle(t.V, t.E);
        expect(a.solver).toBe('direct');
        // MEASURED: every pair of this tangle is near-field.
        expect((a.nearPairs ?? 0) / (300 * 300)).toBeGreaterThan(0.5);

        const g = largeKnot.generate({ vertices: 3000 });
        const b = makeRingSaddle(g.vertices, g.edges);
        expect(b.solver).toBe('gmres');
        expect(directIsCheaper(3000, b.nearPairs ?? 0)).toBe(false);
    });

    test('a forced route is honoured; above the cap the direct route is refused', () => {
        const t = tangle(200, 7);
        expect(makeRingSaddle(t.V, t.E, { linearSolver: 'gmres' }).solver).toBe('gmres');
        expect(makeRingSaddle(t.V, t.E, { linearSolver: 'direct' }).solver).toBe('direct');
        const g = largeKnot.generate({ vertices: DIRECT_MAX_N + 2 });
        expect(makeDirectSaddle(g.vertices, g.edges)).toBeNull();
        expect(makeRingSaddle(g.vertices, g.edges, { linearSolver: 'direct' }).solver).toBe(
            'gmres',
        );
    });

    test('both routes solve the same constrained system: same gradient direction', () => {
        // One barycenter-constrained solve with a smooth right-hand side, tree at a tight
        // tolerance against direct. They differ by the tree's far-field approximation only.
        const g = largeKnot.generate({ vertices: 240 });
        const n = 240;
        const C: number[][] = [0, 1, 2].map((c) => {
            const row = new Array<number>(3 * n).fill(0);
            for (let i = 0; i < n; i++) row[c * n + i] = 1 / n;
            return row;
        });
        const rhs = Array.from({ length: 3 * n }, (_, i) => Math.cos(0.05 * i));
        const solve = (linearSolver: 'gmres' | 'direct') =>
            makeRingSaddle(g.vertices, g.edges, { linearSolver, schur: { tol: 1e-10 } })
                .saddle.prepare(C)
                .solve(rhs).x;
        // MEASURED 2.2e-3: the size of the block cluster tree's own approximation at θ = 0.25.
        expect(relDiff(solve('gmres'), solve('direct'))).toBeLessThan(2e-2);
    });
});

describe("Barnes–Hut differential, traversal 'auto'", () => {
    test('on a tangle it is the exact differential', () => {
        const { V, E } = tangle(200, 8);
        const exact = gradientAnalytical(V, E, calculateDisjointPairs(E), 3, 6, 1e-10).flat();
        const stats = { farNodes: 0, nearPairs: 0 };
        const auto = barnesHutGradient(
            prepareBarnesHut(V, E, 1e-10, { traversal: 'auto' }),
            3,
            6,
            stats,
        ).flat();
        expect(stats.farNodes).toBe(0);
        expect(relDiff(auto, exact)).toBeLessThan(1e-11);
    });

    test('where the tree engages it is the per-vertex walk, unchanged', () => {
        const g = largeKnot.generate({ vertices: 3000 });
        const run = (traversal: 'auto' | 'vertex') => {
            const stats = { farNodes: 0, nearPairs: 0 };
            const grad = barnesHutGradient(
                prepareBarnesHut(g.vertices, g.edges, 1e-10, { traversal }),
                3,
                6,
                stats,
            );
            return { grad, stats };
        };
        const a = run('auto');
        const v = run('vertex');
        expect(a.stats.farNodes).toBeGreaterThan(0);
        expect(a.stats).toEqual(v.stats);
        expect(a.grad).toEqual(v.grad);
    });
});
