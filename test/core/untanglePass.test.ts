// The untangle pass: the medial-sphere flow packaged as a pre-pass for one closed curve.
// @see src/core/untanglePass.ts
import { describe, expect, test } from 'bun:test';
import { knotSignature } from '../../src/core/knotDeterminant';
import { totalLength } from '../../src/core/sobolev/constraintSet';
import type { Edge, Vec3 } from '../../src/core/testConfigs';
import {
    createUntangleRun,
    finishUntangle,
    ringOrder,
    untangleDisplay,
    untangleTick,
} from '../../src/core/untanglePass';

const ring = (n: number): Edge[] => Array.from({ length: n }, (_, i) => [i, (i + 1) % n] as Edge);
function sample(n: number, f: (t: number) => Vec3, amp: number): Vec3[] {
    return Array.from({ length: n }, (_, i) => {
        const t = (2 * Math.PI * i) / n;
        const p = f(t);
        return [
            p[0] + amp * Math.sin(7 * t + 1),
            p[1] + amp * Math.cos(5 * t + 2),
            p[2] + amp * Math.sin(9 * t + 3),
        ] as Vec3;
    });
}
const trefoil = (n: number) =>
    sample(
        n,
        (t) => [
            Math.sin(t) + 2 * Math.sin(2 * t),
            Math.cos(t) - 2 * Math.cos(2 * t),
            -Math.sin(3 * t),
        ],
        0.25,
    );
const unknot = (n: number) => sample(n, (t) => [2 * Math.cos(t), 2 * Math.sin(t), 0], 0.3);
const flat = (V: Vec3[]) => ({ n: V.length, x: Float64Array.from(V.flat()) });
const centre = (V: Vec3[]): Vec3 =>
    V.reduce(
        (a, p) => [a[0] + p[0] / V.length, a[1] + p[1] / V.length, a[2] + p[2] / V.length] as Vec3,
        [0, 0, 0] as Vec3,
    );
function runToEnd(V: Vec3[], E: Edge[]) {
    const run = createUntangleRun(V, E);
    if (!run) throw new Error('not a ring');
    while (run.stopped === null) untangleTick(run, 50, Number.POSITIVE_INFINITY);
    return run;
}

describe('ringOrder', () => {
    test('walks one loop whatever the order and direction of its edges', () => {
        const E: Edge[] = [
            [2, 3],
            [1, 0],
            [3, 0],
            [2, 1],
        ];
        const order = ringOrder(4, E);
        expect(order).not.toBeNull();
        expect([...(order ?? [])].sort()).toEqual([0, 1, 2, 3]);
        // Consecutive entries are joined by an edge.
        const has = (a: number, b: number) =>
            E.some(([u, v]) => (u === a && v === b) || (u === b && v === a));
        for (let k = 0; k < 4; k++)
            expect(has((order ?? [])[k], (order ?? [])[(k + 1) % 4])).toBe(true);
    });
    test('null for an open chain, two loops, and a junction', () => {
        expect(
            ringOrder(4, [
                [0, 1],
                [1, 2],
                [2, 3],
            ]),
        ).toBeNull();
        expect(
            ringOrder(8, [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
            ]),
        ).toBeNull();
        expect(
            ringOrder(4, [
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 2],
            ]),
        ).toBeNull();
    });
});

describe('untangle pass', () => {
    test('a tangled trefoil: same vertex count, length and centre; knot type kept', () => {
        const V = trefoil(600);
        const E = ring(600);
        const run = runToEnd(V, E);
        const res = finishUntangle(run, V);
        expect(res.audit).toBe('kept');
        expect(res.stopped).toBe('plateau');
        expect(res.vertices.length).toBe(600);
        expect(totalLength(res.vertices, E)).toBeCloseTo(totalLength(V, E), 9);
        const c0 = centre(V);
        const c1 = centre(res.vertices);
        for (let q = 0; q < 3; q++) expect(c1[q]).toBeCloseTo(c0[q], 9);
        expect(knotSignature(flat(res.vertices))).toEqual([3, 3]);
    });
    test('an unknot shrinks away and comes back as a round circle', () => {
        const V = unknot(400);
        const E = ring(400);
        const res = finishUntangle(runToEnd(V, E), V);
        expect(res.stopped).toBe('collapsed');
        expect(res.audit).toBe('kept');
        expect(totalLength(res.vertices, E)).toBeCloseTo(totalLength(V, E), 9);
        const c = centre(res.vertices);
        const radii = res.vertices.map((p) => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]));
        expect(Math.max(...radii) - Math.min(...radii)).toBeLessThan(1e-9);
    });
    test('a failed audit returns the input untouched', () => {
        const V = trefoil(600);
        const E = ring(600);
        const run = runToEnd(V, E);
        run.signature0 = [5, 5]; // pretend the input was a figure-eight
        const res = finishUntangle(run, V);
        expect(res.audit).toBe('failed');
        expect(res.vertices).toEqual(V);
    });
    test('the display curve has the input size while the flow shrinks', () => {
        const V = trefoil(600);
        const E = ring(600);
        const run = createUntangleRun(V, E);
        if (!run) throw new Error('not a ring');
        untangleTick(run, 60, Number.POSITIVE_INFINITY);
        expect(run.stopped).toBeNull();
        const out = V.map((p) => [...p] as Vec3);
        untangleDisplay(run, out);
        expect(totalLength(out, E)).toBeCloseTo(totalLength(V, E), 9);
        expect(run.lengths[run.lengths.length - 1]).toBeLessThan(totalLength(V, E));
    });
});
