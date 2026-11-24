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
    test('a right angle at a junction is left alone (no permanent push between its two edges)', () => {
        const V: Vec3[] = [
            [0, 0, 0],
            [2, 0, 0],
            [0, 2, 0],
        ];
        const E: Edge[] = [
            [0, 1],
            [0, 2],
        ];
        const g = medialGraphFrom(V, E, 2 * PIECES);
        const x0 = Float64Array.from(g.x);
        const r = medialGraphStep(g, { r: R, tension: 0 });
        expect(r.links).toBe(0);
        expect(Array.from(g.x)).toEqual(Array.from(x0));
    });

    // One edge of 16 segments, bent by hand: `shape(k)` is the position of its k-th point.
    function bentEdge(shape: (k: number) => [number, number]): MedialGraph {
        const [x0, y0] = shape(0);
        const [x1, y1] = shape(16);
        const g = medialGraphFrom(
            [
                [x0, y0, 0],
                [x1, y1, 0],
            ],
            [[0, 1]],
            16,
        );
        for (let k = 1; k < 16; k++) {
            const [px, py] = shape(k);
            g.x[3 * (1 + k)] = px; // interior points follow the 2 original nodes
            g.x[3 * (1 + k) + 1] = py;
        }
        for (let k = 0; k < 16; k++) {
            const a = g.segs.a[k];
            const b = g.segs.b[k];
            g.rest[k] = Math.hypot(g.x[3 * b] - g.x[3 * a], g.x[3 * b + 1] - g.x[3 * a + 1]);
        }
        return g;
    }
    test('an edge folded back on itself is pushed open; a gentle bend is left alone', () => {
        // Hairpin: up one leg and down the other, the legs 0.06 apart.
        const hairpin = bentEdge((k) => (k <= 8 ? [0, 0.25 * k] : [0.06, 0.25 * (16 - k)]));
        const gap = (g: MedialGraph) =>
            Math.hypot(g.x[3 * 5] - g.x[3 * 13], g.x[3 * 5 + 1] - g.x[3 * 13 + 1]);
        const gap0 = gap(hairpin);
        expect(gap0).toBeCloseTo(0.06, 10);
        run(hairpin, 100);
        expect(gap(hairpin)).toBeGreaterThan(3 * gap0);

        // An arc of radius 3 (far wider than the range): nothing may move.
        const arc = bentEdge((k) => [3 * Math.cos(0.08 * k), 3 * Math.sin(0.08 * k)]);
        const before = Float64Array.from(arc.x);
        // (Bending springs off: they would, rightly, start to straighten the arc.)
        expect(medialGraphStep(arc, { r: R, tension: 0, bend: 0 }).links).toBe(0);
        expect(Array.from(arc.x)).toEqual(Array.from(before));
    });
    test('bending springs: a zigzag edge straightens instead of folding flat', () => {
        // An edge of 16 segments drawn as a zigzag of ±60°, its rest lengths those of the
        // zigzag, so only the bending springs act. Its span must grow, not shrink.
        const zig = bentEdge((k) => [0.1 * k, k % 2 === 0 ? 0 : 0.17]);
        const span = (g: MedialGraph) => Math.hypot(g.x[3] - g.x[0], g.x[4] - g.x[1]);
        const amplitude = (g: MedialGraph) => Math.abs(g.x[3 * 3 + 1] - g.x[3 * 4 + 1]);
        const a0 = amplitude(zig);
        const s0 = span(zig);
        for (let k = 0; k < 60; k++) medialGraphStep(zig, { r: R, tension: 0 });
        expect(amplitude(zig)).toBeLessThan(0.5 * a0);
        expect(span(zig)).toBeGreaterThan(s0);
    });
    test('a dragged node is pulled to its target by a bounded spring and never jams', () => {
        // Bar 2 lies 0.3 above bar 1; drag the middle of bar 1 upwards, THROUGH where bar 2 is.
        const { V, E } = bars(0.3);
        const g = medialGraphFrom(V, E, 4 * PIECES);
        const mid = 4 + 15; // interior point 16 of bar 1 (interior points follow the 4 nodes)
        const target: [number, number] = [g.x[3 * mid], 1.5];
        for (let k = 0; k < 400; k++)
            medialGraphStep(g, { r: R, pinTargets: new Map([[mid, target]]) });
        // It went up, pushing bar 2 ahead of it instead of stopping against it …
        expect(g.x[3 * mid + 1]).toBeGreaterThan(0.6);
        // … bar 2 is still above it everywhere near there, and they are not in contact.
        expect(countPlanarCrossings(vec(g), segs(g), 4 * PIECES)).toBe(0);
        expect(minClearance(g, 2)).toBeGreaterThan(0.05);
    });
    test('a gap that is already tiny cannot close further, however hard it is pressed', () => {
        // Two bars 0.002 apart (under 0.02·r = 0.008 and under 0.02 of a segment = 0.0025);
        // the middle of the upper one is pulled down through the lower one with a force
        // nothing in the flow can match.
        const { V, E } = bars(0.002);
        const g = medialGraphFrom(V, E, 4 * PIECES);
        const mid = 4 + 31 + 15; // interior point 16 of bar 2
        const target = new Map<number, [number, number]>([[mid, [g.x[3 * mid], -2]]]);
        const gap0 = minClearance(g, 1);
        let least = gap0;
        for (let k = 0; k < 200; k++) {
            medialGraphStep(g, { r: R, pinTargets: target, pinForceMax: 1e9, pinStiffness: 1e6 });
            least = Math.min(least, minClearance(g, 1));
        }
        expect(least).toBeGreaterThanOrEqual(gap0 - 1e-12); // not one step nearer
        expect(countPlanarCrossings(vec(g), segs(g), 4 * PIECES)).toBe(0);
        // Left alone, the same bars move apart: the floor holds nothing back that way.
        const free = medialGraphFrom(V, E, 4 * PIECES);
        for (let k = 0; k < 60; k++) medialGraphStep(free, { r: R });
        expect(minClearance(free, 1)).toBeGreaterThan(10 * gap0);
    });
});
