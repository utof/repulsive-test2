/**
 * bench/scale/directVsTree.ts — bun bench/scale/directVsTree.ts [flow.json frame,frame,...]
 *
 * One Sobolev step through `dispatchDescentStep` on the hierarchical route with each linear
 * solver FORCED ('gmres' = block cluster tree + GMRES, 'direct' = exact assembly + Cholesky) and
 * with 'auto', over the same curves:
 *   - smooth trefoils (the tree's case) at several sizes,
 *   - seeded tangles (a random control polygon in a ball, every edge cut into 8 — the shape of
 *     the paper's Knot128 "perturbed" curves; the direct route's case),
 *   - optionally frames of a trajectory recorded by bench/app/recordFlow.ts.
 * Per row: wall time, the step's timing ledger, the near-field fraction of the block lists,
 * what 'auto' chose, and the step length. Appended to bench/results/<date>-direct-vs-tree.jsonl.
 * Why: `directIsCheaper` (src/core/sobolev/hierarchical/directSolve.ts) is a rule with measured
 * constants; this is the measurement, and the way to re-derive them on another machine.
 */
import { appendFileSync, readFileSync } from 'node:fs';
import { dispatchDescentStep } from '../../src/core/dispatch';
import { edgeLengths, totalLength } from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import { buildBct } from '../../src/core/sobolev/hierarchical/bct';
import { buildBvh } from '../../src/core/sobolev/hierarchical/bvh';
import { directIsCheaper } from '../../src/core/sobolev/hierarchical/directSolve';
import { type Edge, testConfigs, type Vec3 } from '../../src/core/testConfigs';

const OUT = `bench/results/${new Date().toISOString().slice(0, 10)}-direct-vs-tree.jsonl`;
const ring = (n: number): Edge[] => Array.from({ length: n }, (_, i) => [i, (i + 1) % n] as Edge);

function tangle(n: number, seed: number): Vec3[] {
    let s = seed;
    const r = () => {
        s = (1664525 * s + 1013904223) % 4294967296;
        return s / 4294967296;
    };
    const control: Vec3[] = [];
    while (control.length < n / 8) {
        const p: Vec3 = [2 * r() - 1, 2 * r() - 1, 2 * r() - 1];
        if (p[0] ** 2 + p[1] ** 2 + p[2] ** 2 <= 1) control.push(p);
    }
    const V: Vec3[] = [];
    control.forEach((a, k) => {
        const b = control[(k + 1) % control.length];
        for (let j = 0; j < 8; j++) {
            const u = j / 8;
            V.push([a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u]);
        }
    });
    return V;
}

const largeKnot = testConfigs.find((t) => t.id === 'knot-large');
if (!largeKnot) throw new Error('the knot-large preset is missing');
const fixtures: { name: string; V: Vec3[] }[] = [];
for (const n of [500, 1000, 2000, 4000])
    fixtures.push({
        name: `smooth trefoil n=${n}`,
        V: largeKnot.generate({ vertices: n }).vertices,
    });
for (const n of [496, 1000, 2000]) fixtures.push({ name: `tangle n=${n}`, V: tangle(n, n) });
if (process.argv[2]) {
    const flow = JSON.parse(readFileSync(process.argv[2], 'utf8'));
    for (const k of (process.argv[3] ?? '0').split(',').map(Number)) {
        const xyz: number[] = flow.frames[k].xyz;
        fixtures.push({
            name: `${process.argv[2].split('/').pop()} frame ${k}`,
            V: Array.from({ length: flow.n }, (_, i) => [
                xyz[3 * i],
                xyz[3 * i + 1],
                xyz[3 * i + 2],
            ]),
        });
    }
}

for (const f of fixtures) {
    const n = f.V.length;
    const edges = ring(n);
    const nearPairs = buildBct(buildBvh(f.V, edges, 8, 1e-10), {}).stats.nearPairs;
    const nearFrac = nearPairs / (n * n);
    for (const solver of ['gmres', 'direct', 'auto'] as const) {
        // The tree route on a 2 000-vertex tangle takes minutes; skip rows that would only say so.
        if (solver === 'gmres' && nearPairs > 2.5e6) {
            console.log(`${f.name} | gmres | skipped (${nearPairs} near pairs)`);
            continue;
        }
        const t0 = performance.now();
        const r = dispatchDescentStep({
            descentMode: 'sobolev',
            vertices: f.V,
            edges,
            disjointPairs: [],
            mode: 'analytical',
            stepSize: 0.001,
            x0: barycenterTarget(f.V, edges),
            sobolevL0: totalLength(f.V, edges),
            sobolevEll0: edgeLengths(f.V, edges),
            barycenterConstraint: true,
            lengthMode: 'total',
            projectionMode: 'frozen',
            sobolevSolver: 'hierarchical',
            hierarchicalLinearSolver: solver,
            collectTimings: true,
        });
        const ms = performance.now() - t0;
        const ran = r.timings?.factor ? 'direct' : 'gmres';
        const row = {
            fixture: f.name,
            n,
            nearFrac,
            ruleSaysDirect: directIsCheaper(n, nearPairs),
            solver,
            ran,
            ms,
            accepted: r.accepted,
            tau: r.stats?.tau ?? null,
            energy: r.energy,
            timings: Object.fromEntries(
                Object.entries(r.timings ?? {}).map(([k, v]) => [k, Math.round(v.ms)]),
            ),
        };
        appendFileSync(OUT, `${JSON.stringify(row)}\n`);
        console.log(
            `${f.name} | ${solver.padEnd(6)} ran ${ran.padEnd(6)} | ${(ms / 1000).toFixed(2)} s | near ${(100 * nearFrac).toFixed(1)}% | tau ${row.tau} | dE ${row.timings.dE} build ${row.timings.assembleA} factor ${row.timings.factor ?? '-'} saddle ${row.timings.saddle}`,
        );
    }
}
