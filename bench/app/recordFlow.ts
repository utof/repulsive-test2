// bench/app/recordFlow.ts — bun bench/app/recordFlow.ts <curve.obj> <steps> <out.json> [solver]
// Runs the app's own step function (`dispatchDescentStep`, Sobolev descent, barycenter + total
// length fixed) on a closed curve read from an OBJ file and writes every iterate to JSON, for
// `renderFlow.py` to turn into a clip. One frame per accepted step.
//
// The OBJ is expected to hold one closed polyline as `v x y z` and `l i j` lines, which is the
// format of the Repulsive Curves knot datasets (Knot128, Trefoil100):
// https://www.cs.cmu.edu/~kmcrane/Projects/RepulsiveCurves/index.html
// The curve is re-indexed along the loop (the hierarchical route needs edges (i, i+1)) and scaled
// to a bounding-box extent of 2, the scale of the app's own presets.
// @see src/core/dispatch.ts, src/core/sobolevRoute.ts
import { readFileSync, writeFileSync } from 'node:fs';
import {
    type DispatchDescentStepArgs,
    dispatchDescentStep,
    graphEnergy,
} from '../../src/core/dispatch';
import { edgeLengths, totalLength } from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import { pairListFits, resolveSobolevRoute, type SobolevSolver } from '../../src/core/sobolevRoute';
import { calculateDisjointPairs } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

const [, , objPath, stepsArg, outPath, solverArg] = process.argv;
if (!objPath || !stepsArg || !outPath) {
    console.error(
        'usage: bun bench/app/recordFlow.ts <curve.obj> <steps> <out.json> [auto|dense|hierarchical]',
    );
    process.exit(2);
}
const steps = Number(stepsArg);
const solver = (solverArg ?? 'auto') as SobolevSolver;

// ---- read one closed polyline
const raw: Vec3[] = [];
const links: [number, number][] = [];
for (const line of readFileSync(objPath, 'utf8').split('\n')) {
    const t = line.trim().split(/\s+/);
    if (t[0] === 'v') raw.push([Number(t[1]), Number(t[2]), Number(t[3])]);
    else if (t[0] === 'l')
        for (let k = 1; k + 1 < t.length; k++) links.push([Number(t[k]) - 1, Number(t[k + 1]) - 1]);
}
const next = new Map<number, number[]>();
for (const [a, b] of links) {
    next.set(a, [...(next.get(a) ?? []), b]);
    next.set(b, [...(next.get(b) ?? []), a]);
}
const order: number[] = [0];
let prev = -1;
let cur = 0;
for (;;) {
    const nb = (next.get(cur) ?? []).filter((x) => x !== prev);
    if (nb.length === 0) break;
    const nxt = nb[0];
    if (nxt === 0) break;
    order.push(nxt);
    prev = cur;
    cur = nxt;
    if (order.length > raw.length) throw new Error('recordFlow: the polyline does not close');
}
if (order.length !== raw.length || links.length !== raw.length)
    throw new Error(
        `recordFlow: expected ONE closed loop; walked ${order.length} of ${raw.length} vertices, ${links.length} edges`,
    );

// ---- re-index along the loop, centre, scale to extent 2
const lo = [Infinity, Infinity, Infinity];
const hi = [-Infinity, -Infinity, -Infinity];
for (const v of raw)
    for (let c = 0; c < 3; c++) {
        lo[c] = Math.min(lo[c], v[c]);
        hi[c] = Math.max(hi[c], v[c]);
    }
const ext = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
const s = 2 / ext;
let V: Vec3[] = order.map((i) => [
    (raw[i][0] - (lo[0] + hi[0]) / 2) * s,
    (raw[i][1] - (lo[1] + hi[1]) / 2) * s,
    (raw[i][2] - (lo[2] + hi[2]) / 2) * s,
]);
const n = V.length;
const edges: Edge[] = V.map((_, i) => [i, (i + 1) % n]);
const dp = pairListFits(n) ? calculateDisjointPairs(edges) : [];

const base = {
    descentMode: 'sobolev',
    edges,
    disjointPairs: dp,
    mode: 'analytical',
    stepSize: 0.001,
    x0: barycenterTarget(V, edges),
    sobolevL0: totalLength(V, edges),
    sobolevEll0: edgeLengths(V, edges),
    barycenterConstraint: true,
    lengthMode: 'total',
    projectionMode: 'frozen',
    sobolevSolver: solver,
} satisfies Omit<DispatchDescentStepArgs, 'vertices'>;

const flat = (W: Vec3[]) =>
    W.flatMap((v) => [Math.fround(v[0]), Math.fround(v[1]), Math.fround(v[2])]);
const frames: { step: number; energy: number; ms: number; tau: number | null; xyz: number[] }[] =
    [];
let energy = graphEnergy(V, edges, dp);
frames.push({ step: 0, energy, ms: 0, tau: null, xyz: flat(V) });
const route = resolveSobolevRoute(solver, n, edges, true);
console.error(`[recordFlow] n=${n} route=${route} E0=${energy.toFixed(4)}`);
let elapsed = 0;
let stopped = 'steps';
// Written every few steps as well as at the end: a tight knot can take a minute per step, and a
// run stopped early should still leave a trajectory behind.
const save = () =>
    writeFileSync(outPath, JSON.stringify({ source: objPath, n, route, solver, stopped, frames }));
for (let k = 1; k <= steps; k++) {
    const t0 = performance.now();
    const r = dispatchDescentStep({ ...base, vertices: V, energyBefore: energy });
    elapsed += performance.now() - t0;
    if (!r.accepted) {
        stopped = r.converged ? 'converged' : (r.stats?.reason ?? 'rejected');
        break;
    }
    V = r.vertices;
    energy = r.energy;
    frames.push({ step: k, energy, ms: elapsed, tau: r.stats?.tau ?? null, xyz: flat(V) });
    if (k % 5 === 0) save();
    if (k % 10 === 0 || elapsed / k > 10_000)
        console.error(
            `[recordFlow] step ${k} E=${energy.toFixed(4)} tau=${r.stats?.tau} ${(elapsed / k).toFixed(0)} ms/step`,
        );
}
save();
console.error(
    `[recordFlow] ${frames.length - 1} steps, stopped: ${stopped}, E ${frames[0].energy.toFixed(3)} -> ${energy.toFixed(3)}, ${(elapsed / 1000).toFixed(1)} s`,
);
