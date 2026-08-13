import { describe, expect, test } from 'bun:test';
import { nearTouchPair, splitHiLo, trefoil } from '../../src/core/fixtures';

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

describe('splitHiLo', () => {
    // Measured (not the originally-targeted 1e-9): the hi/lo split is
    // `lo = fround(c - hi)` — the residual is ROUNDED TO f32, not kept
    // exact (that would be two-sum, which PREC Q3/design §2.3 explicitly
    // rule out as not portable in WGSL). That double-rounding puts an
    // absolute-error floor of ~ulp_f32(residual) ≈ 2^-48 per coordinate
    // (≈3.5e-15 at this O(1) magnitude) under the reconstruction; relative
    // to nearTouchPair's 1e-6 gap that floor is ≈2.5e-9, confirmed by
    // direct measurement (2026-08-13) — still a ~1e5x improvement over the
    // plain-f32 cancellation above (relErr > 1e-3), just not 1e-9 at this
    // specific gap. 1e-8 gives >3x margin over the measured value without
    // papering over the double-rounding floor.
    // @see docs/2026-08-13-ai-research-gpu-precision.md Q1 (extent/gap error law), Q3 (not two-sum)
    test('hi+lo reconstruction recovers the nearTouchPair gap to relErr < 1e-8 (f64)', () => {
        // index 7 = vertex 2's y (3*2+1); index 1 = vertex 0's y (3*0+1) —
        // the same pair the plain-f32 cancellation test above uses, but
        // reconstructed from the hi/lo split instead of hi alone.
        const { vertices, gap } = nearTouchPair(1e-6);
        const { hi, lo } = splitHiLo(vertices);
        const reconstructedGap = hi[7] + lo[7] - (hi[1] + lo[1]);
        const relErr = Math.abs(reconstructedGap - gap) / gap;
        expect(relErr).toBeLessThan(1e-8);
    });
});
