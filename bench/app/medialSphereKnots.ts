// bench/app/medialSphereKnots.ts — bun bench/app/medialSphereKnots.ts <curve.obj> [maxSteps] [budgetSec] [tau] [searchCap] [out.json]
//
// Runs the medial-sphere flow (src/core/medialSphere.ts) on one closed curve from an OBJ file
// (`v` + `l` lines in loop order) to the paper's stopping rule and prints one JSON row.
// Why: the first go/no-go bar of docs/plans/2026-11-24-medial-sphere-method.md.
import { readFileSync, writeFileSync } from 'node:fs';
import { fewestProjectedCrossings, knotSignature } from '../../src/core/knotDeterminant';
import {
    type ClosedCurve,
    curveLength,
    medialSphereScales,
    medialSphereStep,
    resampleClosedCurve,
} from '../../src/core/medialSphere';
import { UNTANGLE_GRADIENT_TOL } from '../../src/core/untanglePass';

const [file, maxStepsArg, budgetArg, tauArg, capArg, outFile] = process.argv.slice(2);
if (!file) {
    console.error(
        'usage: bun bench/app/medialSphereKnots.ts <curve.obj> [maxSteps] [budgetSec] [tau] [searchCap] [out.json]',
    );
    process.exit(2);
}
const maxSteps = Number(maxStepsArg ?? 3000);
const budgetMs = 1000 * Number(budgetArg ?? 120);
const tau = Number(tauArg ?? 1);

const coords: number[] = [];
for (const line of readFileSync(file, 'utf8').split('\n')) {
    if (!line.startsWith('v ')) continue;
    const p = line.trim().split(/\s+/);
    coords.push(Number(p[1]), Number(p[2]), Number(p[3]));
}
let curve: ClosedCurve = { n: coords.length / 3, x: Float64Array.from(coords) };
let lo = Infinity;
let hi = -Infinity;
for (const v of coords) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
}
const searchCap = Number(capArg ?? (hi - lo) / 10);
const scales = medialSphereScales(curve, searchCap);
const n0 = curve.n;
// Independent audit of the knot type: the determinant before, after, and every 25 steps (to
// date a breakage). Not counted in the flow's time.
const sig0 = knotSignature(curve);
// How tangled: fewest crossings over 12 projections, at the start, at the end, and for the
// dataset's tidy prototype of the same knot (Knot128 keeps one in `original/`; a trefoil's is 3).
const crossingsStart = fewestProjectedCrossings(curve);
let crossingsPrototype: number | null = file.includes('/Trefoil100/') ? 3 : null;
if (file.includes('/perturbed/')) {
    const proto: number[] = [];
    for (const line of readFileSync(file.replace('/perturbed/', '/original/'), 'utf8').split(
        '\n',
    )) {
        if (!line.startsWith('v ')) continue;
        const q = line.trim().split(/\s+/);
        proto.push(Number(q[1]), Number(q[2]), Number(q[3]));
    }
    crossingsPrototype = fewestProjectedCrossings({
        n: proto.length / 3,
        x: Float64Array.from(proto),
    });
}
let auditMs = 0;
let brokeAtStep: number | null = null;
const length0 = curveLength(curve);
curve = resampleClosedCurve(curve, scales.h);
const frames: number[][] = [];
const t0 = performance.now();
let steps = 0;
let stopped = 'steps';
let refused = 0;
let zeroRun = 0;
let limitedSteps = 0;
let tryStep = tau;
let cgMax = 0;
let grad = Number.NaN;
const trace: string[] = [];
const lengths: number[] = [];
while (steps < maxSteps) {
    if (performance.now() - t0 - auditMs > budgetMs) {
        stopped = 'time';
        break;
    }
    const r = medialSphereStep(curve, scales, { tau, tryStep });
    tryStep = r.nextTryStep;
    steps++;
    grad = r.gradientPerNode;
    if (r.limited) limitedSteps++;
    cgMax = Math.max(cgMax, r.cgIterations);
    if (r.step === 0) refused++;
    zeroRun = r.step === 0 ? zeroRun + 1 : 0;
    if (zeroRun >= 5) {
        stopped = 'stuck';
        break;
    }
    curve = r.curve;
    if (outFile && steps % 4 === 0) frames.push(Array.from(curve.x));
    if (steps <= 5 || steps % 50 === 0)
        trace.push(
            `step ${steps} n ${curve.n} len ${curveLength(curve).toFixed(2)} grad ${grad.toExponential(2)} tau ${r.step.toExponential(1)} lim ${r.limited ? 1 : 0} cg ${r.cgIterations} links ${r.links} t ${((performance.now() - t0) / 1000).toFixed(1)}s`,
        );
    if (brokeAtStep === null && (steps % 25 === 0 || curve.n <= 10)) {
        const ta = performance.now();
        const sig = knotSignature(curve);
        auditMs += performance.now() - ta;
        if (sig && sig0 && (sig[0] !== sig0[0] || sig[1] !== sig0[1])) brokeAtStep = steps;
    }
    // Plateau rule (ours), a backstop behind the paper's gradient rule below: the shortest
    // length seen in the last 100 steps is within 1% of the shortest in the 100 before.
    lengths.push(curveLength(curve));
    if (steps >= 200) {
        let recent = Infinity;
        let before = Infinity;
        for (let q = steps - 100; q < steps; q++) recent = Math.min(recent, lengths[q]);
        for (let q = steps - 200; q < steps - 100; q++) before = Math.min(before, lengths[q]);
        if (recent > 0.99 * before) {
            stopped = 'plateau';
            break;
        }
    }
    if (curve.n <= 10) {
        stopped = 'collapsed';
        break;
    }
    if (grad < UNTANGLE_GRADIENT_TOL) {
        stopped = 'gradient';
        break;
    }
}
const seconds = (performance.now() - t0 - auditMs) / 1000;
const sigEnd = knotSignature(curve);
if (process.env.TRACE) console.error(trace.join('\n'));
if (outFile) writeFileSync(outFile, JSON.stringify({ file, frames }));
console.log(
    JSON.stringify({
        file: file.split('/').slice(-2).join('/'),
        n0,
        nEnd: curve.n,
        length0: +length0.toFixed(3),
        lengthEnd: +curveLength(curve).toFixed(3),
        r: +scales.r.toFixed(4),
        h: +scales.h.toFixed(4),
        tau,
        steps,
        stopped,
        seconds: +seconds.toFixed(2),
        gradientPerNode: +grad.toExponential(3),
        limitedSteps,
        refusedSteps: refused,
        cgMax,
        crossingsStart,
        crossingsEnd: fewestProjectedCrossings(curve),
        crossingsPrototype,
        determinant0: sig0?.[0] ?? null,
        determinantEnd: sigEnd?.[0] ?? null,
        typeKept:
            sig0 && sigEnd
                ? sig0[0] === sigEnd[0] && sig0[1] === sigEnd[1] && brokeAtStep === null
                : null,
        brokeAtStep,
    }),
);
