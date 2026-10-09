/**
 * Task 1 of Phase 2H: the O(N) cyclic-tridiagonal `(-Delta)^-1`.
 *
 * Bars are MEASURED, not self-anchored: `.superpowers/sdd/2026-09-11-phase2h-queue/scratch/out-tridiag.json`
 * (git-ignored spike artefact), key `rows[].relResidual`, `case == "closed-cyclic(mass=1)"`,
 * Sherman-Morrison, min-of-21-reps, Intel i7-9750H, machine CONTENDED (`host.note`).
 * Measured: n=960 2.86e-13, 3000 1.19e-12, 10000 3.21e-12, 30000 2.12e-11, 100000 1.94e-10.
 * The controller ruling (progress.md; spec §11.2 mass-weight ruling) gates ALL three geometries against
 * those bars because open/circle were only measured at mass=1e-3.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §11.2, §11.3
 */
import { describe, expect, test } from 'bun:test';
import {
    assembleCurveLaplacian,
    CurveTopologyError,
    makeLaplacianWorkspace,
    solveLaplacian,
    triResidual,
} from '../../../src/core/sobolev/hierarchical/laplacian';
import type { Edge, Vec3 } from '../../../src/core/testConfigs';

function chainEdges(n: number, closed: boolean): Edge[] {
    const e: Edge[] = [];
    for (let i = 0; i < n - 1; i++) e.push([i, i + 1]);
    if (closed) e.push([n - 1, 0]);
    return e;
}
function circle(n: number): Vec3[] {
    return Array.from({ length: n }, (_, i) => {
        const t = (2 * Math.PI * i) / n;
        return [Math.cos(t), Math.sin(t), 0] as Vec3;
    });
}
// Same generators as scratch/curves.ts (torusKnot(2,3), openHelix(4 turns)).
function torusKnot(n: number): Vec3[] {
    return Array.from({ length: n }, (_, i) => {
        const t = (2 * Math.PI * i) / n;
        const rad = 2 + 0.8 * Math.cos(3 * t);
        return [rad * Math.cos(2 * t), rad * Math.sin(2 * t), 0.8 * Math.sin(3 * t)] as Vec3;
    });
}
function openHelix(n: number): Vec3[] {
    return Array.from({ length: n }, (_, i) => {
        const t = (4 * 2 * Math.PI * i) / (n - 1);
        return [Math.cos(t), Math.sin(t), (3 * i) / (n - 1)] as Vec3;
    });
}
function rhs(n: number, seed: number): Float64Array {
    let s = seed >>> 0;
    const r = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        r[i] = s / 4294967296 - 0.5;
    }
    return r;
}

function denseOf(sys: ReturnType<typeof assembleCurveLaplacian>): Float64Array[] {
    const { n, a, b, c } = sys;
    const M = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
        M[i][i] += b[i];
        if (i > 0) M[i][i - 1] += a[i];
        if (i < n - 1) M[i][i + 1] += c[i];
    }
    if (sys.closed) {
        M[0][n - 1] += sys.cornerTR;
        M[n - 1][0] += sys.cornerBL;
    }
    return M;
}
function gauss(M: Float64Array[], r: Float64Array): Float64Array {
    const n = r.length;
    const A = M.map((row, i) => Float64Array.from([...row, r[i]]));
    for (let k = 0; k < n; k++) {
        let p = k;
        for (let i = k + 1; i < n; i++) if (Math.abs(A[i][k]) > Math.abs(A[p][k])) p = i;
        [A[k], A[p]] = [A[p], A[k]];
        for (let i = k + 1; i < n; i++) {
            const f = A[i][k] / A[k][k];
            for (let j = k; j <= n; j++) A[i][j] -= f * A[k][j];
        }
    }
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
        let v = A[i][n];
        for (let j = i + 1; j < n; j++) v -= A[i][j] * x[j];
        x[i] = v / A[i][i];
    }
    return x;
}

describe('dense cross-check (proves Sherman-Morrison corner wiring)', () => {
    for (const closed of [true, false]) {
        test(`n=64 ${closed ? 'closed' : 'open'}, mass=1`, () => {
            const n = 64;
            const sys = assembleCurveLaplacian(
                torusKnot(n).map((p, i) => (closed ? p : openHelix(n)[i])),
                chainEdges(n, closed),
            );
            expect(sys.closed).toBe(closed);
            const r = rhs(n, 1);
            const x = new Float64Array(n);
            solveLaplacian(sys, r, x, makeLaplacianWorkspace(n));
            const ref = gauss(denseOf(sys), r);
            let num = 0;
            let den = 0;
            for (let i = 0; i < n; i++) {
                num += (x[i] - ref[i]) ** 2;
                den += ref[i] ** 2;
            }
            expect(Math.sqrt(num / den)).toBeLessThan(1e-12);
        });
    }
});

describe('residual bars (closed-cyclic(mass=1) rows of out-tridiag.json, applied to every geometry)', () => {
    const geoms: Array<[string, (n: number) => Vec3[], boolean]> = [
        ['closed-cyclic (torus knot)', torusKnot, true],
        ['closed-circle', circle, true],
        ['open-helix', openHelix, false],
    ];
    // [n, bar]: <=5e-12 for n<=10000 (measured 2.86e-13..3.21e-12); 5e-11 at 30000 (measured 2.12e-11, spec-named);
    // 5e-10 at 100000 (measured 1.94e-10, spec-named). §11.2 ruling; do NOT loosen on failure - report a finding.
    const bars: Array<[number, number]> = [
        [960, 5e-12],
        [3000, 5e-12],
        [10000, 5e-12],
        [30000, 5e-11],
        [100000, 5e-10],
    ];
    for (const [name, gen, closed] of geoms) {
        for (const [n, bar] of bars) {
            test(`${name} n=${n} <= ${bar}`, () => {
                const sys = assembleCurveLaplacian(gen(n), chainEdges(n, closed));
                const r = rhs(n, 4242);
                const x = new Float64Array(n);
                solveLaplacian(sys, r, x, makeLaplacianWorkspace(n));
                expect(triResidual(sys, x, r)).toBeLessThanOrEqual(bar);
            });
        }
    }
});

describe('structure', () => {
    test('closed = 2 Thomas sweeps, open = 1', () => {
        for (const closed of [true, false]) {
            const n = 50;
            const sys = assembleCurveLaplacian(
                closed ? circle(n) : openHelix(n),
                chainEdges(n, closed),
            );
            const ws = makeLaplacianWorkspace(n);
            solveLaplacian(sys, rhs(n, 2), new Float64Array(n), ws);
            expect(ws.thomasSweeps).toBe(closed ? 2 : 1);
        }
    });
    test('repeat solve with the same workspace is bit-identical (no hidden state)', () => {
        const n = 200;
        const sys = assembleCurveLaplacian(torusKnot(n), chainEdges(n, true));
        const ws = makeLaplacianWorkspace(n);
        const r = rhs(n, 3);
        const x1 = new Float64Array(n);
        const x2 = new Float64Array(n);
        solveLaplacian(sys, r, x1, ws);
        solveLaplacian(sys, r, x2, ws);
        expect(Array.from(x2)).toEqual(Array.from(x1));
    });
});

describe('topology and singularity are named errors', () => {
    test('junction (three edges at a vertex) throws naming the edge', () => {
        const v = circle(4);
        const edges: Edge[] = [
            [0, 1],
            [1, 2],
            [1, 3],
        ];
        expect(() => assembleCurveLaplacian(v, edges)).toThrow(CurveTopologyError);
        expect(() => assembleCurveLaplacian(v, edges)).toThrow(/\[1,3\]/);
    });
    test('permuted-order path throws naming the edge', () => {
        expect(() =>
            assembleCurveLaplacian(circle(3), [
                [0, 2],
                [2, 1],
            ]),
        ).toThrow(/\[0,2\]/);
    });
    test('open chain with a missing edge throws', () => {
        expect(() =>
            assembleCurveLaplacian(circle(4), [
                [0, 1],
                [2, 3],
            ]),
        ).toThrow(CurveTopologyError);
    });
    test('massWeight=0 is singular (closed AND open) and throws', () => {
        expect(() => assembleCurveLaplacian(circle(8), chainEdges(8, true), 1, 0)).toThrow(
            /singular/,
        );
        expect(() => assembleCurveLaplacian(openHelix(8), chainEdges(8, false), 1, 0)).toThrow(
            /singular/,
        );
    });
});

describe('O(N) scaling', () => {
    test('ns/vertex at n=100000 within 4x of n=1000 (measured mass=1 range 25.07-39.36 ns/vtx)', () => {
        const per = (n: number): number => {
            const sys = assembleCurveLaplacian(torusKnot(n), chainEdges(n, true));
            const ws = makeLaplacianWorkspace(n);
            const r = rhs(n, 5);
            const x = new Float64Array(n);
            const reps = n < 10000 ? 400 : 15;
            for (let k = 0; k < 5; k++) solveLaplacian(sys, r, x, ws);
            let best = Infinity;
            for (let k = 0; k < reps; k++) {
                const t0 = performance.now();
                solveLaplacian(sys, r, x, ws);
                best = Math.min(best, performance.now() - t0);
            }
            return (best * 1e6) / n;
        };
        const small = per(1000);
        const big = per(100000);
        console.log(`laplacian ns/vertex: n=1000 ${small.toFixed(2)}, n=100000 ${big.toFixed(2)}`);
        expect(big).toBeLessThan(4 * small);
    });
});
