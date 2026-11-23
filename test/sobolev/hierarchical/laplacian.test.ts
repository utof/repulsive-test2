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
import { fullTest } from '../../tier';

function chainEdges(n: number, closed: boolean): Edge[] {
    const e: Edge[] = [];
    for (let i = 0; i < n - 1; i++) e.push([i, i + 1]);
    if (closed) e.push([n - 1, 0]);
    return e;
}
/**
 * Rescale so the bbox max extent is 1, centred at the origin: verbatim port of `normalize` in
 * scratch/curves.ts:49-66, called by `closedFrom`/`openFrom` (curves.ts:74,84). Omitting it made
 * the first version of these tests use curves 2-5.6x larger (better conditioned) than the artefact's.
 */
function normalize(V: Vec3[]): Vec3[] {
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (const p of V) {
        for (let c = 0; c < 3; c++) {
            if (p[c] < lo[c]) lo[c] = p[c];
            if (p[c] > hi[c]) hi[c] = p[c];
        }
    }
    const ext = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    const s = ext > 0 ? 1 / ext : 1;
    return V.map(
        (p) =>
            [
                (p[0] - 0.5 * (lo[0] + hi[0])) * s,
                (p[1] - 0.5 * (lo[1] + hi[1])) * s,
                (p[2] - 0.5 * (lo[2] + hi[2])) * s,
            ] as Vec3,
    );
}
function circle(n: number): Vec3[] {
    return normalize(
        Array.from({ length: n }, (_, i) => {
            const t = (2 * Math.PI * i) / n;
            return [Math.cos(t), Math.sin(t), 0] as Vec3;
        }),
    );
}
// Same generators AND normalization as scratch/curves.ts (torusKnot(2,3), openHelix(4 turns)).
function torusKnot(n: number): Vec3[] {
    return normalize(
        Array.from({ length: n }, (_, i) => {
            const t = (2 * Math.PI * i) / n;
            const rad = 2 + 0.8 * Math.cos(3 * t);
            return [rad * Math.cos(2 * t), rad * Math.sin(2 * t), 0.8 * Math.sin(3 * t)] as Vec3;
        }),
    );
}
function openHelix(n: number): Vec3[] {
    return normalize(
        Array.from({ length: n }, (_, i) => {
            const t = (4 * 2 * Math.PI * i) / (n - 1);
            return [Math.cos(t), Math.sin(t), (3 * i) / (n - 1)] as Vec3;
        }),
    );
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
                closed ? torusKnot(n) : openHelix(n),
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

function residualOf(gen: (n: number) => Vec3[], closed: boolean, n: number, seed: number): number {
    const sys = assembleCurveLaplacian(gen(n), chainEdges(n, closed));
    const r = rhs(n, seed);
    const x = new Float64Array(n);
    solveLaplacian(sys, r, x, makeLaplacianWorkspace(n));
    return triResidual(sys, x, r);
}

const NS = [960, 3000, 10000, 30000, 100000];

describe('check 1, fidelity: closed-cyclic reproduces out-tridiag.json to 4 significant digits', () => {
    // scratch/out-tridiag.json rows[] case "closed-cyclic(mass=1)", key relResidual, geometry torusKnot(2,3)
    // (normalized), rhs seed 4242, mass=1, Sherman-Morrison; the residual is a property of the solve, not the
    // timing, so the contended host / min-of-21 do not affect it. This is the port-fidelity oracle.
    const ref: Record<number, number> = {
        960: 2.8647e-13,
        3000: 1.1903e-12,
        10000: 3.2063e-12,
        30000: 2.1214e-11,
        100000: 1.9401e-10,
    };
    for (const n of NS) {
        test(`torus knot n=${n} == ${ref[n]}`, () => {
            const got = residualOf(torusKnot, true, n, 4242);
            expect(Math.abs(got - ref[n]) / ref[n]).toBeLessThan(5e-4);
        });
    }
});

describe('check 2, stability: max over 3 seeds <= 1000 * N * u_f64, every geometry', () => {
    // Floor N*u_f64 is DERIVED (backward stability of an O(N) sweep). The x1000 is CHOSEN headroom, justified
    // by two MEASURED facts (normalized curves, mass=1, this branch's first review, /tmp/lap-scale-check.ts):
    //  - seed spread 5.4x: circle n=30000 gives 1.43e-10 at rhs seed 31337 vs 2.66e-11 at 4242;
    //  - worst constant vs the floor: c ~ 43 circle vs 6.4 knot at n=30000 (c = residual / (N*u_f64)).
    // It CATCHES a corner sign error, a wrong gamma, or a dropped mass term (orders of magnitude). It does NOT
    // certify accuracy: that is check 1's job. Replaces the controller's first ruling (one geometry's bars
    // transferred to two others), which was self-anchoring (spec §7.5/§10.2) and which circle breaches.
    const U = 1.1102230246251565e-16;
    const geoms: Array<[string, (n: number) => Vec3[], boolean]> = [
        ['torus knot', torusKnot, true],
        ['circle', circle, true],
        ['open helix', openHelix, false],
    ];
    for (const [name, gen, closed] of geoms) {
        for (const n of NS) {
            const bar = 1000 * n * U; // 1.07e-10 (960) .. 1.11e-8 (100000)
            test(`${name} n=${n} <= ${bar.toExponential(2)}`, () => {
                const worst = Math.max(
                    ...[4242, 777, 31337].map((s) => residualOf(gen, closed, n, s)),
                );
                expect(worst).toBeLessThanOrEqual(bar);
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
        ).toThrow(/missing edge \[1,2\]/);
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
    // FULL TIER ONLY (test/tier.ts). Why: a wall-clock ratio, i.e. a load-dependent instrument
    // (the default runner puts four test files on the CPU at once); best-of-N protects it and it
    // passed under 4-way load, but a per-merge run should not hinge on load. An O(N) -> O(N log N)
    // regression in `solveLaplacian` is now seen at the next full run (README.md "Tests").
    fullTest(
        'ns/vertex at n=100000 within 4x of n=1000 (measured mass=1 range 25.07-73.01 ns/vtx incl. the n=960 row; 4x is slack)',
        () => {
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
            expect(big).toBeLessThan(4 * small);
        },
    );
});
