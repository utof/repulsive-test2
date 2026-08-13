import { describe, expect, test } from 'bun:test';
import { nearTouchPair, trefoil } from '../../src/core/fixtures';

describe('trefoil', () => {
    test('produces n vertices and n closed-loop edges', () => {
        const { vertices, edges } = trefoil(60);
        expect(vertices.length).toBe(60);
        expect(edges.length).toBe(60);
        expect(edges[59]).toEqual([59, 0]);
    });
    test('is deterministic', () => {
        expect(trefoil(120)).toEqual(trefoil(120));
    });
    test('matches the bench parametrization at i=1, n=4 exactly', () => {
        // p(t) = (sin t + 2 sin 2t, cos t − 2 cos 2t, −sin 3t), t = 2πi/n
        const t = (2 * Math.PI) / 4;
        const [x, y, z] = trefoil(4).vertices[1];
        expect(x).toBe(Math.sin(t) + 2 * Math.sin(2 * t));
        expect(y).toBe(Math.cos(t) - 2 * Math.cos(2 * t));
        expect(z).toBe(-Math.sin(3 * t));
    });
});

describe('nearTouchPair', () => {
    test('two disjoint edges separated by the requested gap', () => {
        const { vertices, edges, gap } = nearTouchPair(1e-6);
        expect(vertices.length).toBe(4);
        expect(edges).toEqual([
            [0, 1],
            [2, 3],
        ]);
        expect(gap).toBe(1e-6);
        // closest approach between the two parallel segments is the gap (f64)
        expect(vertices[2][1] - vertices[0][1]).toBeCloseTo(1e-6, 12);
    });
    test('f32 storage destroys the gap — the cancellation property T1/G2 need (review F1)', () => {
        // The whole point of this fixture: coordinates are O(1) values NOT
        // exactly representable in f32, so rounding the positions to f32
        // corrupts the 1e-6 gap by orders of magnitude. If this assertion
        // fails, the fixture is NOT stressing cancellation and every gate
        // built on it is vacuous. @see PREC Q1 (error law ~ β·u_f·(extent/gap))
        const { vertices } = nearTouchPair(1e-6);
        const d32 = Math.fround(vertices[2][1]) - Math.fround(vertices[0][1]);
        const relErr = Math.abs(d32 - 1e-6) / 1e-6;
        expect(relErr).toBeGreaterThan(1e-3);
    });
});
