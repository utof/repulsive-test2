// bench/app/medialLocalDrag.ts — bun bench/app/medialLocalDrag.ts [size=60] [settle=150] [drag=200] [sleep=on|off|<threshold>] [out]
// The case the mind-map product lives in: a LARGE drawing at rest, and one node dragged a
// short way. Asks what a step costs then, with and without leaving the resting part of the
// drawing alone (`sleep` in `medialGraphStep`).
//   fixture   a jittered size×size grid, 2 points per edge, no tension, settled for `settle` steps
//   drag      the middle node is pulled 2 units diagonally over `drag` steps (bounded spring)
// One JSON line: time per step while settling and while dragging, how many nodes were awake,
// how far the dragged node got, crossings, and — against the same run WITHOUT sleeping — the
// largest difference in any node's final position (`worstVsAwake`, in units of r): what
// leaving the rest alone costs in accuracy.
// @see docs/plans/2026-11-24-medial-sphere-method.md ("Only step what moves")
import { appendFileSync } from 'node:fs';
import { countCrossings } from '../../src/core/medialGraphMove';
import {
    type MedialGraph,
    medialGraphFrom,
    medialGraphStep,
    minClearance,
} from '../../src/core/medialSphereGraph';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

const SIZE = Number(process.argv[2] ?? 60);
const SETTLE = Number(process.argv[3] ?? 150);
const DRAG = Number(process.argv[4] ?? 200);
const SLEEP = (process.argv[5] ?? 'on') !== 'off';
// `on`, `off`, or the threshold itself (a node sleeps when it moves less than this·r a step).
const BELOW = Number.isFinite(Number(process.argv[5])) ? Number(process.argv[5]) : undefined;
const OUT = process.argv[6] ?? 'bench/results/2026-11-24-local-drag.jsonl';
const R = 0.4;

function grid(size: number): MedialGraph {
    let s = 7;
    const rnd = (): number => {
        s = (1664525 * s + 1013904223) % 4294967296;
        return s / 4294967296;
    };
    const V: Vec3[] = [];
    const E: Edge[] = [];
    for (let y = 0; y < size; y++)
        for (let x = 0; x < size; x++)
            V.push([x + 0.3 * (rnd() - 0.5), y + 0.3 * (rnd() - 0.5), 0]);
    for (let y = 0; y < size; y++)
        for (let x = 0; x < size; x++) {
            if (x + 1 < size) E.push([y * size + x, y * size + x + 1]);
            if (y + 1 < size) E.push([y * size + x, (y + 1) * size + x]);
        }
    return medialGraphFrom(V, E, 2);
}

/** Settle, then drag; returns the graph and the measurements. */
function run(sleep: boolean) {
    const g = grid(SIZE);
    // No tension: with it a grid drawn at its rest lengths shrinks by an eighth as a whole,
    // a motion of the entire drawing that takes thousands of steps at this size — the
    // fixture would never be "at rest", which is the case this benchmark is about.
    const opts = sleep ? { r: R, tension: 0, sleep: { below: BELOW } } : { r: R, tension: 0 };
    let t = performance.now();
    let awakeSettleEnd = g.n;
    let moveSettleEnd = 0;
    for (let k = 0; k < SETTLE; k++) {
        const st = medialGraphStep(g, opts);
        awakeSettleEnd = st.awakeNodes;
        moveSettleEnd = st.movePerNode;
    }
    const msSettle = (performance.now() - t) / Math.max(1, SETTLE);
    // What a step costs once the drawing is at rest (20 more steps, nothing touched).
    t = performance.now();
    for (let k = 0; k < 20; k++) medialGraphStep(g, opts);
    const msAtRest = (performance.now() - t) / 20;
    const mid = Math.floor(SIZE / 2) * SIZE + Math.floor(SIZE / 2);
    const from: [number, number] = [g.x[3 * mid], g.x[3 * mid + 1]];
    const to: [number, number] = [from[0] + 2, from[1] + 2];
    const before = Float64Array.from(g.x);
    t = performance.now();
    let awakeSum = 0;
    let awakeMax = 0;
    for (let k = 1; k <= DRAG; k++) {
        const f = Math.min(1, (2 * k) / DRAG); // target arrives halfway, then holds
        const target = new Map<number, [number, number]>([
            [mid, [from[0] + 2 * f, from[1] + 2 * f]],
        ]);
        const st = medialGraphStep(g, { ...opts, pinTargets: target });
        awakeSum += st.awakeNodes;
        awakeMax = Math.max(awakeMax, st.awakeNodes);
    }
    const msDrag = (performance.now() - t) / Math.max(1, DRAG);
    let moved = 0;
    for (let i = 0; i < g.n; i++)
        if (Math.hypot(g.x[3 * i] - before[3 * i], g.x[3 * i + 1] - before[3 * i + 1]) > 0.01 * R)
            moved++;
    return {
        g,
        msSettle,
        msAtRest,
        msDrag,
        awakeSettleEnd,
        moveSettleEnd,
        awakeMean: awakeSum / Math.max(1, DRAG),
        awakeMax,
        moved,
        fromTarget: Math.hypot(g.x[3 * mid] - to[0], g.x[3 * mid + 1] - to[1]),
    };
}

const a = run(SLEEP);
let worstVsAwake: number | null = null;
let awakeRun: { msAtRest: number; msDrag: number; moved: number } | null = null;
if (SLEEP) {
    const b = run(false);
    awakeRun = b;
    worstVsAwake = 0;
    for (let i = 0; i < a.g.n; i++)
        worstVsAwake = Math.max(
            worstVsAwake,
            Math.hypot(a.g.x[3 * i] - b.g.x[3 * i], a.g.x[3 * i + 1] - b.g.x[3 * i + 1]) / R,
        );
}
const row = {
    bench: 'medial-local-drag',
    size: SIZE,
    nodes: a.g.n,
    segments: a.g.segs.a.length,
    r: R,
    sleep: SLEEP,
    sleepBelow: SLEEP ? (BELOW ?? 2e-4) : null,
    settleSteps: SETTLE,
    dragSteps: DRAG,
    msPerStepSettling: Number(a.msSettle.toFixed(2)),
    msPerStepAtRest: Number(a.msAtRest.toFixed(2)),
    msPerStepDragging: Number(a.msDrag.toFixed(2)),
    msPerStepAtRestAllAwake: awakeRun ? Number(awakeRun.msAtRest.toFixed(2)) : null,
    msPerStepDraggingAllAwake: awakeRun ? Number(awakeRun.msDrag.toFixed(2)) : null,
    nodesMovedByDragAllAwake: awakeRun ? awakeRun.moved : null,
    awakeAfterSettling: a.awakeSettleEnd,
    movePerNodeAfterSettling: Number(a.moveSettleEnd.toExponential(2)),
    awakeMeanDragging: Math.round(a.awakeMean),
    awakeMaxDragging: a.awakeMax,
    nodesMovedByDrag: a.moved,
    draggedFromTarget: Number(a.fromTarget.toFixed(3)),
    crossingsEnd: countCrossings(a.g),
    clearanceEnd: Number(minClearance(a.g, 2 * R).toFixed(3)),
    worstVsAwake: worstVsAwake === null ? null : Number(worstVsAwake.toFixed(4)),
    finite: a.g.x.every(Number.isFinite),
};
console.log(JSON.stringify(row));
appendFileSync(OUT, `${JSON.stringify(row)}\n`);
