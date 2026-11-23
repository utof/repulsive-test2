/**
 * bench/scale/graphRoutes.ts — bun bench/scale/graphRoutes.ts
 *
 * One Sobolev step on GRAPHS that are not one closed loop (3-D grids and binary trees, laid out
 * by the force layout first, like the app after its pre-pass), through `dispatchDescentStep` on
 * the dense route and on the hierarchical route — which for such graphs is the direct solver.
 * Per row: wall time, the timing ledger, the step length, and the exact-energy decrease (the two
 * routes must agree). Appended to bench/results/<date>-graph-routes.jsonl.
 * Why: `GRAPH_AUTO_MIN_N` (src/core/sobolevRoute.ts) is set from these numbers, and they show
 * where a graph step's time goes once the solve is cheap (the all-pairs differential).
 */
import { appendFileSync } from 'node:fs';
import { dispatchDescentStep } from '../../src/core/dispatch';
import { runForceLayout } from '../../src/core/forceLayout';
import { edgeLengths, totalLength } from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import { calculateDisjointPairs, calculateEnergy } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

const OUT = `bench/results/${new Date().toISOString().slice(0, 10)}-graph-routes.jsonl`;
/** The dense route factors a (3n+k)² matrix; above this it only measures patience. */
const DENSE_MAX_N = 600;

function rng(seed: number): () => number {
    let s = seed;
    return () => {
        s = (1664525 * s + 1013904223) % 4294967296;
        return s / 4294967296;
    };
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
const tree = (n: number): Edge[] =>
    Array.from({ length: n - 1 }, (_, i) => [Math.floor(i / 2), i + 1] as Edge);

const fixtures: { name: string; n: number; edges: Edge[] }[] = [
    { name: 'tree 31', n: 31, edges: tree(31) },
    { name: 'grid 3^3', n: 27, edges: grid3(3) },
    { name: 'grid 4^3', n: 64, edges: grid3(4) },
    { name: 'tree 127', n: 127, edges: tree(127) },
    { name: 'grid 6^3', n: 216, edges: grid3(6) },
    { name: 'tree 511', n: 511, edges: tree(511) },
    { name: 'grid 8^3', n: 512, edges: grid3(8) },
    { name: 'grid 10^3', n: 1000, edges: grid3(10) },
    { name: 'tree 2047', n: 2047, edges: tree(2047) },
];
for (const [k, f] of fixtures.entries()) {
    const r = rng(k + 1);
    const start: Vec3[] = Array.from({ length: f.n }, () => [
        4 * r() - 2,
        4 * r() - 2,
        4 * r() - 2,
    ]);
    const V = runForceLayout(start, f.edges).vertices;
    const dense = f.n <= DENSE_MAX_N;
    const pairs = dense ? calculateDisjointPairs(f.edges) : [];
    const exact = (W: Vec3[]) =>
        dense ? calculateEnergy(W, f.edges, pairs, 3, 6, 1e-10) : Number.NaN;
    const e0 = exact(V);
    for (const solver of ['dense', 'hierarchical'] as const) {
        if (solver === 'dense' && !dense) continue;
        const t0 = performance.now();
        const res = dispatchDescentStep({
            descentMode: 'sobolev',
            vertices: V,
            edges: f.edges,
            disjointPairs: pairs,
            mode: 'analytical',
            stepSize: 0.001,
            x0: barycenterTarget(V, f.edges),
            sobolevL0: totalLength(V, f.edges),
            sobolevEll0: edgeLengths(V, f.edges),
            barycenterConstraint: true,
            lengthMode: 'total',
            projectionMode: 'frozen',
            sobolevSolver: solver,
            collectTimings: true,
        });
        const ms = performance.now() - t0;
        const row = {
            fixture: f.name,
            nodes: f.n,
            edges: f.edges.length,
            solver,
            direct: res.timings?.factor !== undefined && solver === 'hierarchical',
            ms,
            accepted: res.accepted,
            tau: res.stats?.tau ?? null,
            exactDecrease: e0 - exact(res.vertices),
            timings: Object.fromEntries(
                Object.entries(res.timings ?? {}).map(([key, v]) => [key, Math.round(v.ms)]),
            ),
        };
        appendFileSync(OUT, `${JSON.stringify(row)}\n`);
        console.log(
            `${f.name} (${f.n} nodes, ${f.edges.length} edges) | ${solver.padEnd(12)} | ${ms.toFixed(0)} ms | tau ${row.tau} | decrease ${row.exactDecrease.toPrecision(10)} | dE ${row.timings.dE} assemble ${row.timings.assembleA} factor ${row.timings.factor ?? '-'}`,
        );
    }
}
