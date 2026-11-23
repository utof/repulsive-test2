/**
 * bench/app/untangleHandoff.ts — bun bench/app/untangleHandoff.ts <curve.obj> [steps=10] [budgetSec=120] [raw|skipraw]
 *
 * The hand-off the app's "untangle first" switch makes: run the untangle pass
 * (src/core/untanglePass.ts) on a dataset knot, then give the result — same vertex count,
 * length and centre — to the tangent-point descent (`dispatchDescentStep`, solver 'auto') for
 * a few steps; and, for comparison, the same number of tangent-point steps from the RAW tangle.
 * Measured: time of the pass, knot-type audit, time per tangent-point step and the energy,
 * before and after. One JSON line is appended to bench/results/<date>-untangle-handoff.jsonl.
 * @see docs/plans/2026-11-24-medial-sphere-method.md
 */
import { appendFileSync, readFileSync } from 'node:fs';
import { dispatchDescentStep, graphEnergy } from '../../src/core/dispatch';
import { edgeLengths, totalLength } from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import { pairListFits } from '../../src/core/sobolevRoute';
import { calculateDisjointPairs } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';
import { createUntangleRun, finishUntangle, untangleTick } from '../../src/core/untanglePass';

const [file, stepsArg, budgetArg, rawArg] = process.argv.slice(2);
if (!file) {
    console.error(
        'usage: bun bench/app/untangleHandoff.ts <curve.obj> [steps=10] [budgetSec=120] [raw|skipraw]',
    );
    process.exit(2);
}
const steps = Number(stepsArg ?? 10);
const budgetMs = 1000 * Number(budgetArg ?? 120);
const OUT = `bench/results/${new Date().toISOString().slice(0, 10)}-untangle-handoff.jsonl`;

// The dataset stores a loop as `v` lines in order plus `l i i+1` lines; scale to extent 2, as
// the app's presets are.
const raw: Vec3[] = [];
for (const line of readFileSync(file, 'utf8').split('\n')) {
    if (!line.startsWith('v ')) continue;
    const p = line.trim().split(/\s+/);
    raw.push([Number(p[1]), Number(p[2]), Number(p[3])]);
}
let lo = Infinity;
let hi = -Infinity;
for (const v of raw)
    for (const c of v) {
        lo = Math.min(lo, c);
        hi = Math.max(hi, c);
    }
const V0: Vec3[] = raw.map((v) => [
    (2 * v[0]) / (hi - lo),
    (2 * v[1]) / (hi - lo),
    (2 * v[2]) / (hi - lo),
]);
const n = V0.length;
const edges: Edge[] = V0.map((_, i) => [i, (i + 1) % n]);
const pairs = pairListFits(n) ? calculateDisjointPairs(edges) : [];

function descend(start: Vec3[]): {
    steps: number;
    msPerStep: number | null;
    e0: number;
    eEnd: number;
    stopped: string;
} {
    let V = start;
    const base = {
        descentMode: 'sobolev' as const,
        edges,
        disjointPairs: pairs,
        mode: 'analytical' as const,
        stepSize: 0.001,
        x0: barycenterTarget(V, edges),
        sobolevL0: totalLength(V, edges),
        sobolevEll0: edgeLengths(V, edges),
        barycenterConstraint: true,
        lengthMode: 'total' as const,
        projectionMode: 'frozen' as const,
        sobolevSolver: 'auto' as const,
    };
    const e0 = graphEnergy(V, edges, pairs);
    let energy = e0;
    let ms = 0;
    let k = 0;
    let stopped = 'steps';
    while (k < steps) {
        if (ms > budgetMs) {
            stopped = 'time';
            break;
        }
        const t = performance.now();
        const r = dispatchDescentStep({ ...base, vertices: V, energyBefore: energy });
        ms += performance.now() - t;
        if (!r.accepted) {
            stopped = r.converged ? 'converged' : (r.stats?.reason ?? 'rejected');
            break;
        }
        V = r.vertices;
        energy = r.energy;
        k++;
    }
    return {
        steps: k,
        msPerStep: k > 0 ? +(ms / k).toFixed(1) : null,
        e0: +e0.toPrecision(6),
        eEnd: +energy.toPrecision(6),
        stopped,
    };
}

const t0 = performance.now();
const run = createUntangleRun(V0, edges);
if (!run) throw new Error('untangleHandoff: not a single closed loop');
const createMs = performance.now() - t0;
const t1 = performance.now();
while (run.stopped === null) untangleTick(run, 50, 1e9);
const passMs = performance.now() - t1;
const t2 = performance.now();
const result = finishUntangle(run, V0);
const finishMs = performance.now() - t2;

const row = {
    file: file.split('/').slice(-2).join('/'),
    n,
    pass: {
        steps: result.steps,
        stopped: result.stopped,
        audit: result.audit,
        seconds: +(passMs / 1000).toFixed(2),
        startupSeconds: +(createMs / 1000).toFixed(2),
        finishSeconds: +(finishMs / 1000).toFixed(2),
        lengthKept: +(totalLength(result.vertices, edges) / totalLength(V0, edges)).toFixed(6),
    },
    afterPass: descend(result.vertices),
    raw: rawArg === 'skipraw' ? null : descend(V0),
};
appendFileSync(OUT, `${JSON.stringify(row)}\n`);
console.log(JSON.stringify(row));
