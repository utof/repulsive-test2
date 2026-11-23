/**
 * The untangle pass: the medial-sphere flow (src/core/medialSphere.ts) packaged as a pre-pass
 * for ONE closed curve, the way src/core/forceLayout.ts is a pre-pass for graphs. It takes a
 * tangled loop, runs the flow until the knot is tight (or, for an unknot, until it has shrunk
 * away), and hands back a curve with the SAME vertex count, total length and centre as the one
 * it was given, so the tangent-point descent can take over with its targets re-anchored.
 *
 * Why a pre-pass and not a descent mode: the medial-sphere flow shrinks the curve and changes
 * its vertex count every step, has no pins and no constraints; the tangent-point flow is slow
 * exactly on tight tangles (96% of pairs near-field). Each does the part the other cannot.
 *
 * The knot type is audited end to end with an invariant that shares no code with the flow's
 * collision test (`knotSignature`): the result is refused if it changed.
 * @see docs/plans/2026-11-24-medial-sphere-method.md
 */
import { knotSignature } from './knotDeterminant';
import {
    type ClosedCurve,
    curveLength,
    type MedialSphereScales,
    medialSphereScales,
    medialSphereStep,
    resampleClosedCurve,
} from './medialSphere';
import type { Edge, Vec3 } from './testConfigs';

/** How a pass ended. */
export type UntangleStop = 'collapsed' | 'plateau' | 'steps';

/** State of one untangle pass. */
export interface UntangleRun {
    /** The flow's current curve (its own vertex count, shrinking). */
    curve: ClosedCurve;
    scales: MedialSphereScales;
    tryStep: number | undefined;
    /** Graph vertex index of the k-th point along the loop, for writing results back. */
    order: number[];
    /** Total length and centre of the input, restored at the end. */
    length0: number;
    center: Vec3;
    /** Knot determinant of the input (null: the projection had too many crossings to audit). */
    signature0: [number, number] | null;
    steps: number;
    /** Curve length after every step (the plateau rule reads it). */
    lengths: number[];
    stopped: UntangleStop | null;
}

/** Outcome of {@link finishUntangle}. */
export interface UntangleResult {
    /** New positions, indexed like the graph's vertices. */
    vertices: Vec3[];
    stopped: UntangleStop;
    steps: number;
    /** `kept`: determinant unchanged. `unverified`: could not be computed. `failed`: it
     *  changed — `vertices` is then the INPUT, untouched. */
    audit: 'kept' | 'unverified' | 'failed';
}

/** Most steps a pass may take (the dataset knots need 200–360). */
export const UNTANGLE_MAX_STEPS = 5000;

/**
 * The vertices of a graph in order along its single cycle, or `null` when the graph is not
 * exactly one closed loop through every vertex (every degree 2, connected).
 */
export function ringOrder(vertexCount: number, edges: readonly Edge[]): number[] | null {
    if (vertexCount < 4 || edges.length !== vertexCount) return null;
    const a = new Int32Array(vertexCount).fill(-1);
    const b = new Int32Array(vertexCount).fill(-1);
    for (const [u, v] of edges) {
        for (const [p, q] of [
            [u, v],
            [v, u],
        ]) {
            if (a[p] < 0) a[p] = q;
            else if (b[p] < 0) b[p] = q;
            else return null;
        }
    }
    const order = [0];
    let prev = -1;
    let cur = 0;
    for (let k = 1; k < vertexCount; k++) {
        if (a[cur] < 0 || b[cur] < 0) return null;
        const next = a[cur] !== prev ? a[cur] : b[cur];
        prev = cur;
        cur = next;
        if (cur === 0) return null; // closed early: more than one loop
        order.push(cur);
    }
    return a[cur] === 0 || b[cur] === 0 ? order : null;
}

/** Start a pass on a graph that is one closed loop; `null` if it is not. */
export function createUntangleRun(
    vertices: readonly Vec3[],
    edges: readonly Edge[],
): UntangleRun | null {
    const order = ringOrder(vertices.length, edges);
    if (!order) return null;
    const n = order.length;
    const x = new Float64Array(3 * n);
    const center: Vec3 = [0, 0, 0];
    let lo = Infinity;
    let hi = -Infinity;
    for (let k = 0; k < n; k++) {
        const p = vertices[order[k]];
        for (let c = 0; c < 3; c++) {
            x[3 * k + c] = p[c];
            center[c] += p[c] / n;
            lo = Math.min(lo, p[c]);
            hi = Math.max(hi, p[c]);
        }
    }
    const input: ClosedCurve = { n, x };
    // Search radius for the start-up scale: a tenth of the bounding box, as in the benchmark.
    const scales = medialSphereScales(input, (hi - lo) / 10 || 1);
    return {
        curve: resampleClosedCurve(input, scales.h),
        scales,
        tryStep: undefined,
        order,
        length0: curveLength(input),
        center,
        signature0: knotSignature(input),
        steps: 0,
        lengths: [],
        stopped: null,
    };
}

/** Advance a pass by at most `maxSteps` steps or `budgetMs` milliseconds. */
export function untangleTick(run: UntangleRun, maxSteps: number, budgetMs: number): void {
    const t0 = performance.now();
    for (let k = 0; k < maxSteps && run.stopped === null; k++) {
        const r = medialSphereStep(run.curve, run.scales, { tryStep: run.tryStep });
        run.curve = r.curve;
        run.tryStep = r.nextTryStep;
        run.steps++;
        run.lengths.push(curveLength(run.curve));
        if (run.curve.n <= 10) run.stopped = 'collapsed';
        else if (run.steps >= UNTANGLE_MAX_STEPS) run.stopped = 'steps';
        else if (run.steps >= 200) {
            // Plateau: the shortest length of the last 100 steps is within 1% of the shortest
            // of the 100 before. Why not the paper's gradient rule: our gradient chatters once
            // the knot is tight and never gets under its threshold (see the plan's results).
            let recent = Infinity;
            let before = Infinity;
            for (let q = run.steps - 100; q < run.steps; q++)
                recent = Math.min(recent, run.lengths[q]);
            for (let q = run.steps - 200; q < run.steps - 100; q++)
                before = Math.min(before, run.lengths[q]);
            if (recent > 0.99 * before) run.stopped = 'plateau';
        }
        if (performance.now() - t0 > budgetMs) break;
    }
}

/**
 * `count` points evenly spaced by arc length along a closed curve, scaled to total length
 * `length` and centred at `center`. The points lie ON the curve's polyline, so the polygon
 * through them cuts each corner by less than one output spacing.
 */
function resampleUniform(
    c: ClosedCurve,
    count: number,
    length: number,
    center: Vec3,
): Float64Array {
    const { n, x } = c;
    const cum = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        cum[i + 1] =
            cum[i] +
            Math.hypot(
                x[3 * j] - x[3 * i],
                x[3 * j + 1] - x[3 * i + 1],
                x[3 * j + 2] - x[3 * i + 2],
            );
    }
    const total = cum[n] || 1;
    const out = new Float64Array(3 * count);
    const mean = [0, 0, 0];
    let seg = 0;
    for (let k = 0; k < count; k++) {
        const s = (k / count) * total;
        while (seg < n - 1 && cum[seg + 1] <= s) seg++;
        const span = cum[seg + 1] - cum[seg];
        const u = span > 0 ? (s - cum[seg]) / span : 0;
        const j = (seg + 1) % n;
        for (let q = 0; q < 3; q++) {
            out[3 * k + q] = x[3 * seg + q] + u * (x[3 * j + q] - x[3 * seg + q]);
            mean[q] += out[3 * k + q] / count;
        }
    }
    // Scale by the length of the OUTPUT polygon, so the result has exactly `length`.
    const outLen = curveLength({ n: count, x: out }) || 1;
    const scale = length / outLen;
    for (let k = 0; k < count; k++)
        for (let q = 0; q < 3; q++) out[3 * k + q] = center[q] + scale * (out[3 * k + q] - mean[q]);
    return out;
}

/**
 * Write the pass's current curve into `out` (the graph's vertex array, mutated in place) at
 * the input's size and centre. For DISPLAY while the pass runs: the flow's own curve shrinks
 * and would be a dot on screen, so — as the paper does for its figures — it is shown rescaled.
 */
export function untangleDisplay(run: UntangleRun, out: Vec3[]): void {
    const pts = resampleUniform(run.curve, run.order.length, run.length0, run.center);
    for (let k = 0; k < run.order.length; k++) {
        const p = out[run.order[k]];
        p[0] = pts[3 * k];
        p[1] = pts[3 * k + 1];
        p[2] = pts[3 * k + 2];
    }
}

/**
 * End a pass: the curve at the input's vertex count, length and centre, with the knot type
 * audited against the input. A curve that shrank away (an unknot) becomes a round circle, as
 * in the paper. If the audit fails, `fallback` (the input positions) is returned unchanged.
 */
export function finishUntangle(run: UntangleRun, fallback: readonly Vec3[]): UntangleResult {
    const n = run.order.length;
    const stopped = run.stopped ?? 'steps';
    let pts: Float64Array;
    if (stopped === 'collapsed') {
        const radius = run.length0 / (2 * n * Math.sin(Math.PI / n));
        pts = new Float64Array(3 * n);
        for (let k = 0; k < n; k++) {
            pts[3 * k] = run.center[0] + radius * Math.cos((2 * Math.PI * k) / n);
            pts[3 * k + 1] = run.center[1] + radius * Math.sin((2 * Math.PI * k) / n);
            pts[3 * k + 2] = run.center[2];
        }
    } else pts = resampleUniform(run.curve, n, run.length0, run.center);
    const sig = knotSignature({ n, x: pts });
    const audit =
        run.signature0 === null || sig === null
            ? 'unverified'
            : sig[0] === run.signature0[0] && sig[1] === run.signature0[1]
              ? 'kept'
              : 'failed';
    const vertices: Vec3[] = fallback.map((p) => [p[0], p[1], p[2]]);
    if (audit !== 'failed')
        for (let k = 0; k < n; k++)
            vertices[run.order[k]] = [pts[3 * k], pts[3 * k + 1], pts[3 * k + 2]];
    return { vertices, stopped, steps: run.steps, audit };
}
