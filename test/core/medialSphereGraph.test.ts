// The flat-graph version of the medial-sphere flow (an experiment — see the module header).
// @see docs/plans/2026-11-24-medial-sphere-method.md
import { describe, expect, test } from 'bun:test';
import {
    type MedialGraph,
    medialGraphFrom,
    medialGraphStep,
    minClearance,
} from '../../src/core/medialSphereGraph';
import { countPlanarCrossings } from '../../src/core/subdivide';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

const R = 0.4;
const PIECES = 8;
const vec = (g: MedialGraph): Vec3[] =>
    Array.from({ length: g.n }, (_, i) => [g.x[3 * i], g.x[3 * i + 1], 0] as Vec3);
const segs = (g: MedialGraph): Edge[] => Array.from(g.segs.a, (a, k) => [a, g.segs.b[k]] as Edge);
function run(g: MedialGraph, steps: number): number {
    let move = Number.NaN;
    for (let k = 0; k < steps; k++) {
        const r = medialGraphStep(g, { r: R });
        move = r.movePerNode;
        if (move < 3e-4) break;
    }
    return move;
}
/** Two parallel bars of length 4, `gap` apart. */
const bars = (gap: number): { V: Vec3[]; E: Edge[] } => ({
    V: [
        [0, 0, 0],
        [4, 0, 0],
        [0, gap, 0],
        [4, gap, 0],
    ],
    E: [
        [0, 1],
        [2, 3],
    ],
});

describe('medialGraphStep', () => {
    test('two bars closer than the range move apart; their lengths are kept', () => {
        const { V, E } = bars(0.2);
        const g = medialGraphFrom(V, E, 4 * PIECES);
        const before = minClearance(g, 2);
        run(g, 200);
        const after = minClearance(g, 2);
        expect(before).toBeCloseTo(0.2, 10);
        expect(after).toBeGreaterThan(2 * before);
        for (const [a, b] of E)
            expect(
                Math.hypot(g.x[3 * b] - g.x[3 * a], g.x[3 * b + 1] - g.x[3 * a + 1]),
            ).toBeCloseTo(4, 0);
    });
    test('two bars farther apart than the range do not push each other', () => {
        const { V, E } = bars(1.5);
        const g = medialGraphFrom(V, E, 4 * PIECES);
        const x0 = Float64Array.from(g.x);
        const r = medialGraphStep(g, { r: R });
        expect(r.links).toBe(0);
        // The only force left is the tension along each bar: nothing moves sideways.
        for (let i = 0; i < g.n; i++) expect(g.x[3 * i + 1]).toBe(x0[3 * i + 1]);
        // And with the tension off, nothing moves at all.
        const h = medialGraphFrom(V, E, 4 * PIECES);
        expect(medialGraphStep(h, { r: R, tension: 0 }).movePerNode).toBe(0);
        expect(Array.from(h.x)).toEqual(Array.from(x0));
    });
    test('a pinned bar stays exactly where it is while the other makes room', () => {
        const { V, E } = bars(0.2);
        const g = medialGraphFrom(V, E, 4 * PIECES);
        // Pin the whole first bar: its two end nodes and its interior points.
        const interior = 4 * PIECES - 1;
        g.pinned[0] = 1;
        g.pinned[1] = 1;
        for (let q = 0; q < interior; q++) g.pinned[4 + q] = 1;
        const x0 = Float64Array.from(g.x);
        run(g, 200);
        for (let i = 0; i < g.n; i++)
            if (g.pinned[i]) {
                expect(g.x[3 * i]).toBe(x0[3 * i]);
                expect(g.x[3 * i + 1]).toBe(x0[3 * i + 1]);
            }
        expect(minClearance(g, 2)).toBeGreaterThan(0.4);
    });
    test('a star with two rays nearly on top of each other opens the angle; no crossing appears', () => {
        // Junction at the origin, three rays; two of them 6° apart.
        const a = (deg: number): Vec3 => [
            2 * Math.cos((deg * Math.PI) / 180),
            2 * Math.sin((deg * Math.PI) / 180),
            0,
        ];
        const V: Vec3[] = [[0, 0, 0], a(0), a(6), a(180)];
        const E: Edge[] = [
            [0, 1],
            [0, 2],
            [0, 3],
        ];
        const g = medialGraphFrom(V, E, 2 * PIECES);
        const angle = (): number =>
            Math.abs(
                Math.atan2(g.x[3 * 2 + 1], g.x[3 * 2]) - Math.atan2(g.x[3 * 1 + 1], g.x[3 * 1]),
            ) *
            (180 / Math.PI);
        expect(angle()).toBeCloseTo(6, 6);
        run(g, 300);
        expect(angle()).toBeGreaterThan(12);
        expect(countPlanarCrossings(vec(g), segs(g), 2 * PIECES)).toBe(0);
    });
});
