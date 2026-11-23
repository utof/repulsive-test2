/**
 * Graphs that are not one closed loop on the hierarchical route, where they run the DIRECT
 * solver: junctions, open chains, several components. The reference is the dense route, which
 * has always served them; every energy compared is the exact all-pairs energy of the output.
 * @see src/core/sobolev/hierarchical/directSolve.ts, src/core/sobolevRoute.ts (GRAPH_AUTO_MIN_N)
 */
import { describe, expect, setDefaultTimeout, test } from 'bun:test';
import { type DispatchDescentStepArgs, dispatchDescentStep } from '../src/core/dispatch';
import { runForceLayout } from '../src/core/forceLayout';
import { DEFAULTS } from '../src/core/optimizer';
import {
    edgeLengths,
    edgeLengthsBlock,
    evaluateConstraintSet,
    totalLength,
} from '../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../src/core/sobolev/constraints';
import { makeDirectSolve } from '../src/core/sobolev/hierarchical/directSolve';
import {
    HierarchicalSaddleUnsupportedError,
    makeAutoSaddle,
} from '../src/core/sobolev/hierarchical/hierarchicalSaddle';
import { SchurConvergenceError } from '../src/core/sobolev/hierarchical/schur';
import { calculateDisjointPairs, calculateEnergy } from '../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../src/core/testConfigs';

setDefaultTimeout(120_000);
const { alpha: ALPHA, beta: BETA, epsilon: EPS } = DEFAULTS;

function lcg(seed: number): () => number {
    let s = seed;
    return () => {
        s = (1664525 * s + 1013904223) % 4294967296;
        return s / 4294967296;
    };
}
/** Random positions, then the force layout: a spread-out, well-conditioned start. */
function laidOut(n: number, edges: Edge[], seed: number): Vec3[] {
    const r = lcg(seed);
    const V: Vec3[] = Array.from({ length: n }, () => [4 * r() - 2, 4 * r() - 2, 4 * r() - 2]);
    return runForceLayout(V, edges).vertices;
}
function grid3(g: number): Edge[] {
    const E: Edge[] = [];
    const id = (x: number, y: number, z: number) => (z * g + y) * g + x;
    for (let z = 0; z < g; z++)
        for (let y = 0; y < g; y++)
            for (let x = 0; x < g; x++) {
                if (x + 1 < g) E.push([id(x, y, z), id(x + 1, y, z)]);
                if (y + 1 < g) E.push([id(x, y, z), id(x, y + 1, z)]);
                if (z + 1 < g) E.push([id(x, y, z), id(x, y, z + 1)]);
            }
    return E;
}
const binaryTree = (n: number): Edge[] =>
    Array.from({ length: n - 1 }, (_, i) => [Math.floor(i / 2), i + 1] as Edge);
const chain = (n: number): Edge[] => Array.from({ length: n - 1 }, (_, i) => [i, i + 1] as Edge);
/** Two separate closed loops of `k` vertices each. */
const twoRings = (k: number): Edge[] => [
    ...Array.from({ length: k }, (_, i) => [i, (i + 1) % k] as Edge),
    ...Array.from({ length: k }, (_, i) => [k + i, k + ((i + 1) % k)] as Edge),
];

function argsFor(vertices: Vec3[], edges: Edge[], disjointPairs: number[][]) {
    return {
        descentMode: 'sobolev',
        vertices,
        edges,
        disjointPairs,
        mode: 'analytical',
        stepSize: 0.001,
        x0: barycenterTarget(vertices, edges),
        sobolevL0: totalLength(vertices, edges),
        sobolevEll0: edgeLengths(vertices, edges),
        barycenterConstraint: true,
        lengthMode: 'total',
        projectionMode: 'frozen',
    } satisfies DispatchDescentStepArgs;
}

const FIXTURES: { name: string; n: number; edges: Edge[] }[] = [
    { name: '4x4x4 grid (junctions of degree 3-6)', n: 64, edges: grid3(4) },
    { name: 'binary tree, 127 nodes', n: 127, edges: binaryTree(127) },
    { name: 'open chain, 90 nodes', n: 90, edges: chain(90) },
    { name: 'two separate rings, 2 x 40', n: 80, edges: twoRings(40) },
];

describe('a graph on the hierarchical route takes the dense route’s step', () => {
    FIXTURES.forEach((f, k) => {
        const V = laidOut(f.n, f.edges, 11 + k);
        const dp = calculateDisjointPairs(f.edges);
        const exact = (W: Vec3[]) => calculateEnergy(W, f.edges, dp, ALPHA, BETA, EPS);
        const base = argsFor(V, f.edges, dp);
        const e0 = exact(V);

        test(`${f.name}: barycenter + total length`, () => {
            const d = dispatchDescentStep({ ...base, sobolevSolver: 'dense' });
            const h = dispatchDescentStep({
                ...base,
                sobolevSolver: 'hierarchical',
                collectTimings: true,
            });
            expect(d.accepted).toBe(true);
            expect(h.accepted).toBe(true);
            expect(h.timings?.factor?.calls).toBe(1); // the direct solver ran
            expect(h.stats?.tau).toBe(d.stats?.tau);
            const dD = e0 - exact(d.vertices);
            const dH = e0 - exact(h.vertices);
            expect(dD).toBeGreaterThan(0);
            // MEASURED 4e-15 … 6e-14 on graphs of 27 to 512 nodes.
            expect(Math.abs(dH - dD) / dD).toBeLessThan(1e-9);
        });

        test(`${f.name}: with three pinned vertices`, () => {
            const pins = [0, 7, f.n - 1].map((vertexIndex) => ({
                vertexIndex,
                target: [...V[vertexIndex]] as Vec3,
                enabled: true,
            }));
            const d = dispatchDescentStep({ ...base, pins, sobolevSolver: 'dense' });
            const h = dispatchDescentStep({ ...base, pins, sobolevSolver: 'hierarchical' });
            expect(h.accepted).toBe(d.accepted);
            const dD = e0 - exact(d.vertices);
            const dH = e0 - exact(h.vertices);
            expect(Math.abs(dH - dD)).toBeLessThan(1e-9 * Math.max(Math.abs(dD), 1e-12) + 1e-12);
            for (const p of pins) {
                const q = h.vertices[p.vertexIndex];
                expect(
                    Math.hypot(q[0] - p.target[0], q[1] - p.target[1], q[2] - p.target[2]),
                ).toBeLessThan(1e-10);
            }
        });

        test(`${f.name}: per-edge lengths go to the dense route, bit for bit`, () => {
            const d = dispatchDescentStep({
                ...base,
                lengthMode: 'perEdge',
                sobolevSolver: 'dense',
            });
            const h = dispatchDescentStep({
                ...base,
                lengthMode: 'perEdge',
                sobolevSolver: 'hierarchical',
            });
            expect(h.vertices).toEqual(d.vertices);
            expect(h.energy).toBe(d.energy);
        });
    });
});

describe('where the direct route does not apply, it says so', () => {
    test('a star has no two disjoint edges: the factorization is refused', () => {
        const edges: Edge[] = Array.from({ length: 8 }, (_, i) => [0, i + 1] as Edge);
        const V = laidOut(9, edges, 3);
        expect(makeDirectSolve(V, edges, ALPHA, BETA, EPS)).toBeNull();
        // …and the step falls back to the dense route, which reports the singular system.
        const base = argsFor(V, edges, calculateDisjointPairs(edges));
        const h = dispatchDescentStep({ ...base, sobolevSolver: 'hierarchical' });
        const d = dispatchDescentStep({ ...base, sobolevSolver: 'dense' });
        expect(h.accepted).toBe(d.accepted);
        expect(h.stats?.reason).toBe(d.stats?.reason);
    });

    test('an isolated vertex is refused the same way', () => {
        const edges = chain(30);
        const V = [...laidOut(30, edges, 4), [9, 9, 9] as Vec3];
        expect(makeDirectSolve(V, edges, ALPHA, BETA, EPS)).toBeNull();
    });

    test('a graph has no tree route, and no per-edge solver', () => {
        const edges = grid3(3);
        const V = laidOut(27, edges, 5);
        expect(() => makeAutoSaddle(V, edges, { linearSolver: 'gmres' })).toThrow(
            HierarchicalSaddleUnsupportedError,
        );
        const { saddle, solver } = makeAutoSaddle(V, edges);
        expect(solver).toBe('direct');
        // The |E| length rows of a graph: recognised and refused, not mis-solved.
        const { C } = evaluateConstraintSet([edgeLengthsBlock(edgeLengths(V, edges))], V, edges);
        expect(() => saddle.prepare(C)).toThrow(HierarchicalSaddleUnsupportedError);
    });
});

describe('SchurOptions.exactInverse', () => {
    test('one application per solve, and an inexact "inverse" is refused, not believed', () => {
        const edges = grid3(3);
        const V = laidOut(27, edges, 6);
        const n = 27;
        const C: number[][] = [0, 1, 2].map((c) => {
            const row = new Array<number>(3 * n).fill(0);
            for (let i = 0; i < n; i++) row[c * n + i] = 1 / n;
            return row;
        });
        const { saddle } = makeAutoSaddle(V, edges);
        const solver = saddle.prepare(C);
        if (!('columns' in solver.report)) throw new Error('expected the Schur route');
        for (const col of solver.report.columns) {
            expect(col.iters).toBe(0);
            expect(col.precondApplies).toBe(1);
            expect(col.eta).toBeLessThan(1e-10);
        }
        // A deliberately wrong inverse (half of it) under a strict certificate must throw.
        const d = makeDirectSolve(V, edges, ALPHA, BETA, EPS);
        if (!d) throw new Error('the factorization failed');
        const half = {
            n,
            sigma: d.precond.sigma,
            apply: (R: ArrayLike<number>, out: Float64Array) => {
                d.precond.apply(R, out);
                for (let i = 0; i < out.length; i++) out[i] *= 0.5;
            },
        };
        const { prepareSchur } = require('../src/core/sobolev/hierarchical/schur');
        expect(() => prepareSchur(d.op, half, C, { exactInverse: true, etaAccept: 1e-8 })).toThrow(
            SchurConvergenceError,
        );
    });
});
