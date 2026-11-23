// The force-directed pre-pass (src/core/forceLayout.ts): d3-force's forces and schedule.
import { describe, expect, test } from 'bun:test';
import {
    createForceLayout,
    forceLayoutSettled,
    forceLayoutTick,
    runForceLayout,
} from '../../src/core/forceLayout';
import { calculateDisjointPairs, calculateEnergy } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

function lcg(seed: number): () => number {
    let s = seed;
    return () => {
        s = (1664525 * s + 1013904223) % 4294967296;
        return s / 4294967296;
    };
}
const len = (V: Vec3[], [a, b]: Edge) =>
    Math.hypot(V[a][0] - V[b][0], V[a][1] - V[b][1], V[a][2] - V[b][2]);
const total = (V: Vec3[], E: Edge[]) => E.reduce((s, e) => s + len(V, e), 0);

/** g×g grid graph, nodes thrown at random into a square (planar) or a cube. */
function scrambledGrid(g: number, seed: number, planar: boolean): { V: Vec3[]; E: Edge[] } {
    const r = lcg(seed);
    const V: Vec3[] = [];
    const E: Edge[] = [];
    for (let i = 0; i < g * g; i++) V.push([4 * r() - 2, 4 * r() - 2, planar ? 0 : 4 * r() - 2]);
    for (let y = 0; y < g; y++)
        for (let x = 0; x < g; x++) {
            if (x + 1 < g) E.push([y * g + x, y * g + x + 1]);
            if (y + 1 < g) E.push([y * g + x, (y + 1) * g + x]);
        }
    return { V, E };
}
function crossings(V: Vec3[], E: Edge[]): number {
    const o = (a: Vec3, b: Vec3, c: Vec3) =>
        Math.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
    let n = 0;
    for (let i = 0; i < E.length; i++)
        for (let j = i + 1; j < E.length; j++) {
            const [a, b] = E[i];
            const [c, d] = E[j];
            if (a === c || a === d || b === c || b === d) continue;
            if (
                o(V[a], V[b], V[c]) * o(V[a], V[b], V[d]) < 0 &&
                o(V[c], V[d], V[a]) * o(V[c], V[d], V[b]) < 0
            )
                n++;
        }
    return n;
}

describe('forceLayout', () => {
    test("the schedule is d3's: alpha 1 → below 0.001 in exactly 300 ticks", () => {
        const { V, E } = scrambledGrid(4, 1, false);
        const s = createForceLayout(V, E);
        expect(s.alpha).toBe(1);
        for (let k = 0; k < 299; k++) forceLayoutTick(s);
        expect(forceLayoutSettled(s)).toBe(false);
        forceLayoutTick(s);
        expect(forceLayoutSettled(s)).toBe(true);
        expect(s.ticks).toBe(300);
    });

    test('a single edge relaxes to the rest length; two unconnected nodes repel', () => {
        const spring = createForceLayout(
            [
                [0, 0, 0],
                [5, 0, 0],
            ],
            [[0, 1]],
            { linkDistance: 1, chargeStrength: 0 },
        );
        while (!forceLayoutSettled(spring)) forceLayoutTick(spring);
        expect(Math.abs(spring.x[3] - spring.x[0])).toBeCloseTo(1, 2);

        const pair = createForceLayout(
            [
                [0, 0, 0],
                [1, 0, 0],
            ],
            [],
            { linkDistance: 1 },
        );
        forceLayoutTick(pair);
        expect(pair.x[3] - pair.x[0]).toBeGreaterThan(1);
    });

    test('it is deterministic, does not mutate its input, and keeps the vertex mean', () => {
        const { V, E } = scrambledGrid(5, 2, false);
        const copy = V.map((p) => [...p] as Vec3);
        const a = runForceLayout(V, E);
        const b = runForceLayout(V, E);
        expect(V).toEqual(copy);
        expect(a.vertices).toEqual(b.vertices);
        for (let c = 0; c < 3; c++) {
            const mean = (W: Vec3[]) => W.reduce((s, p) => s + p[c], 0) / W.length;
            expect(mean(a.vertices)).toBeCloseTo(mean(V), 9);
        }
    });

    test('runForceLayout keeps the total edge length of the input', () => {
        const { V, E } = scrambledGrid(5, 3, false);
        expect(total(runForceLayout(V, E).vertices, E)).toBeCloseTo(total(V, E), 9);
    });

    test('the result does not depend on the scale of the input', () => {
        const { V, E } = scrambledGrid(4, 4, false);
        const big = V.map((p) => [1000 * p[0], 1000 * p[1], 1000 * p[2]] as Vec3);
        const a = runForceLayout(V, E).vertices;
        const b = runForceLayout(big, E).vertices;
        for (let i = 0; i < a.length; i++)
            for (let c = 0; c < 3; c++) expect(b[i][c] / 1000).toBeCloseTo(a[i][c], 6);
    });

    test('a planar input stays planar, and coincident nodes separate', () => {
        const { V, E } = scrambledGrid(4, 5, true);
        V[3] = [...V[2]] as Vec3; // two nodes on the same spot
        const out = runForceLayout(V, E).vertices;
        for (const p of out) expect(p[2]).toBe(0);
        expect(Math.hypot(out[3][0] - out[2][0], out[3][1] - out[2][1])).toBeGreaterThan(1e-3);
        for (const p of out) for (const c of p) expect(Number.isFinite(c)).toBe(true);
    });

    test('pinned vertices do not move', () => {
        const { V, E } = scrambledGrid(4, 6, false);
        const s = createForceLayout(V, E, { fixed: new Set([0, 7]) });
        while (!forceLayoutSettled(s)) forceLayoutTick(s);
        for (const i of [0, 7]) for (let c = 0; c < 3; c++) expect(s.x[3 * i + c]).toBe(V[i][c]);
        expect(s.x[3]).not.toBe(V[1][0]);
    });

    test('it untangles: a scrambled planar grid loses most crossings and most energy', () => {
        // MEASURED (bench/app/forcePrePass.ts, 2026-11-23): 10x10 grid 3 724 → 54 crossings,
        // tangent-point energy 1.24e6 → 1.07e3. The bars below leave room for other seeds.
        const { V, E } = scrambledGrid(8, 7, true);
        const out = runForceLayout(V, E).vertices;
        const pairs = calculateDisjointPairs(E);
        const energy = (W: Vec3[]) => calculateEnergy(W, E, pairs, 3, 6, 1e-10);
        expect(crossings(out, E)).toBeLessThan(crossings(V, E) / 10);
        expect(energy(out)).toBeLessThan(energy(V) / 20);
    });
});
