/**
 * bench/app/subdivisionCrossings.ts — bun bench/app/subdivisionCrossings.ts [maxSteps=120] [maxSeconds=60]
 *
 * Does cutting every edge into several segments stop the flow from passing edges through each
 * other? Flat graphs (z = 0), each stepped with the app's own `dispatchDescentStep` (Sobolev
 * descent, barycenter + total length, solver 'auto') at 1 to 8 segments per edge, without and
 * with the line search's crossing guard (`preventCrossings`, src/core/planarCollision.ts). In the
 * plane the number of edge crossings can only change when two edges pass through each other, so
 * the crossing count along the run IS the measurement: it is counted after every step, and a
 * run "holds" when it never changes.
 * Fixtures: a clean jittered grid (0 crossings), and graphs laid out by the force layout from a
 * scramble (a few crossings left) — the state the app is in after its pre-pass.
 * One JSON line per (fixture, pieces) is appended to bench/results/<date>-subdivision-crossings.jsonl.
 * @see src/core/subdivide.ts
 */
import { appendFileSync } from 'node:fs';
import { dispatchDescentStep } from '../../src/core/dispatch';
import { runForceLayout } from '../../src/core/forceLayout';
import { edgeLengths, totalLength } from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import { pairListFits } from '../../src/core/sobolevRoute';
import { countPlanarCrossings, subdivideGraph } from '../../src/core/subdivide';
import { calculateDisjointPairs } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

const maxSteps = Number(process.argv[2] ?? 120);
const maxSeconds = Number(process.argv[3] ?? 60);
const OUT = `bench/results/${new Date().toISOString().slice(0, 10)}-subdivision-crossings.jsonl`;

function rng(seed: number): () => number {
    let s = seed;
    return () => {
        s = (1664525 * s + 1013904223) % 4294967296;
        return s / 4294967296;
    };
}
function grid2(g: number): Edge[] {
    const E: Edge[] = [];
    for (let y = 0; y < g; y++)
        for (let x = 0; x < g; x++) {
            if (x + 1 < g) E.push([y * g + x, y * g + x + 1]);
            if (y + 1 < g) E.push([y * g + x, (y + 1) * g + x]);
        }
    return E;
}
const tree = (n: number): Edge[] =>
    Array.from({ length: n - 1 }, (_, i) => [Math.floor(i / 2), i + 1] as Edge);
const scrambled = (n: number, seed: number): Vec3[] => {
    const r = rng(seed);
    return Array.from({ length: n }, () => [4 * r() - 2, 4 * r() - 2, 0] as Vec3);
};
const cleanGrid = (g: number, seed: number): Vec3[] => {
    const r = rng(seed);
    const V: Vec3[] = [];
    for (let y = 0; y < g; y++)
        for (let x = 0; x < g; x++) V.push([x + 0.6 * (r() - 0.5), y + 0.6 * (r() - 0.5), 0]);
    return V;
};

const fixtures: { name: string; V: Vec3[]; E: Edge[] }[] = [
    { name: '6x6 grid, clean (jittered lattice)', V: cleanGrid(6, 1), E: grid2(6) },
    {
        name: '6x6 grid, force layout of a scramble',
        V: runForceLayout(scrambled(36, 1), grid2(6)).vertices,
        E: grid2(6),
    },
    {
        name: '10x10 grid, force layout of a scramble',
        V: runForceLayout(scrambled(100, 2), grid2(10)).vertices,
        E: grid2(10),
    },
    {
        name: 'binary tree 63, force layout of a scramble',
        V: runForceLayout(scrambled(63, 3), tree(63)).vertices,
        E: tree(63),
    },
    { name: '6x6 grid, scrambled (no pre-pass)', V: scrambled(36, 1), E: grid2(6) },
];

for (const f of fixtures) {
    for (const [pieces, guard] of [
        [1, false],
        [4, false],
        [8, false],
        [1, true],
        [2, true],
        [4, true],
        [8, true],
    ] as const) {
        const g = subdivideGraph(f.V, f.E, pieces);
        const pairs = pairListFits(g.vertices.length) ? calculateDisjointPairs(g.edges) : [];
        const crossings = (V: Vec3[]) => countPlanarCrossings(V, g.edges, pieces);
        const c0 = crossings(g.vertices);
        let V = g.vertices;
        let energy: number | undefined;
        let steps = 0;
        let stopped = 'steps';
        let firstChange: number | null = null;
        let cMin = c0;
        let cMax = c0;
        let e0: number | null = null;
        const base = {
            descentMode: 'sobolev' as const,
            edges: g.edges,
            disjointPairs: pairs,
            mode: 'analytical' as const,
            stepSize: 0.001,
            x0: barycenterTarget(V, g.edges),
            sobolevL0: totalLength(V, g.edges),
            sobolevEll0: edgeLengths(V, g.edges),
            barycenterConstraint: true,
            lengthMode: 'total' as const,
            projectionMode: 'frozen' as const,
            sobolevSolver: 'auto' as const,
            preventCrossings: guard,
        };
        const t0 = performance.now();
        let ms = 0;
        while (steps < maxSteps && ms < maxSeconds * 1000) {
            const t = performance.now();
            const r = dispatchDescentStep({
                ...base,
                vertices: V,
                ...(energy === undefined ? {} : { energyBefore: energy }),
            });
            ms += performance.now() - t;
            if (!r.accepted) {
                stopped = r.converged ? 'converged' : (r.stats?.reason ?? 'rejected');
                break;
            }
            V = r.vertices;
            energy = r.energy;
            e0 ??= r.energy;
            steps++;
            const c = crossings(V);
            cMin = Math.min(cMin, c);
            cMax = Math.max(cMax, c);
            if (c !== c0 && firstChange === null) firstChange = steps;
        }
        if (stopped === 'steps' && steps < maxSteps) stopped = 'time';
        const cEnd = crossings(V);
        const maxZ = Math.max(...V.map((p) => Math.abs(p[2])));
        const row = {
            fixture: f.name,
            pieces,
            guard,
            vertices: g.vertices.length,
            segments: g.edges.length,
            crossingsStart: c0,
            crossingsEnd: cEnd,
            crossingsMin: cMin,
            crossingsMax: cMax,
            firstChangeAtStep: firstChange,
            held: firstChange === null,
            steps,
            stopped,
            msPerStep: steps > 0 ? ms / steps : null,
            energyAfterFirstStep: e0,
            energyEnd: energy ?? null,
            maxAbsZ: maxZ,
            wallSeconds: (performance.now() - t0) / 1000,
        };
        appendFileSync(OUT, `${JSON.stringify(row)}\n`);
        console.log(
            `${f.name} | x${pieces} ${guard ? 'guard' : 'plain'} (${row.vertices} pts) | crossings ${c0} -> ${cEnd} [${cMin}..${cMax}] ${row.held ? 'HELD' : `changed at step ${firstChange}`} | ${steps} steps, ${stopped}, ${row.msPerStep?.toFixed(0)} ms/step | E ${energy?.toPrecision(5)} | max|z| ${maxZ.toExponential(0)}`,
        );
    }
}
