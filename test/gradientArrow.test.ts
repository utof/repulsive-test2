import { expect, test } from 'bun:test';
import type { Vec3 } from '../src/core/testConfigs';
import { projectArrow } from '../src/viewRotation';

// Regression tests for gradient-arrow rendering. Two distinct bugs are covered:
//
//  A. "arrows change direction when viewing from a different angle": arrows were drawn
//     at a FIXED screen length along a normalized 2D direction, so they never
//     foreshortened and whipped around when a gradient pointed toward the camera.
//  B. "arrows skew crazily at extreme/intersecting points": projecting the far arrow
//     endpoint drove the tip toward the perspective near-plane (s = 3/(z2+6)) for the
//     huge gradients near intersections, so s exploded/flipped and the arrow shot
//     off-canvas.
//
// projectArrow fixes both: local projection derivative for direction+foreshortening
// (well-conditioned), magnitude carried as a clamped scalar. @see src/viewRotation.ts

const W = 800;
const H = 600;
const zoom = 1;
const SCALE = 0.2;
const BIG = 1e9; // effectively-unbounded maxLen for tests that aren't about the cap

test('A: arrows FORESHORTEN as the view turns a fixed gradient toward the camera', () => {
    const vertex: Vec3 = [0, 0, 0];
    const negGrad: Vec3 = [0, 0, 1]; // world +Z
    const mag = 1;

    const broadside = projectArrow(
        vertex,
        negGrad,
        mag,
        W,
        H,
        Math.PI / 2,
        0,
        zoom,
        SCALE,
        0,
        BIG,
    ).len;
    const turned = projectArrow(
        vertex,
        negGrad,
        mag,
        W,
        H,
        Math.PI / 4,
        0,
        zoom,
        SCALE,
        0,
        BIG,
    ).len;

    expect(broadside).toBeGreaterThan(0);
    // Old fixed-length rendering gave ratio ~1.0 (no foreshortening); true projection is well below.
    expect(turned / broadside).toBeLessThan(0.85);
});

test('min-length floor keeps a tiny-magnitude gradient visible', () => {
    const vertex: Vec3 = [0, 0, 0];
    const tiny: Vec3 = [0.01, 0.01, 0]; // in-plane so it has a well-defined direction
    const mag = Math.hypot(tiny[0], tiny[1]);
    const MIN = 14;

    const withFloor = projectArrow(vertex, tiny, mag, W, H, 0, 0, zoom, SCALE, MIN, BIG).len;
    const noFloor = projectArrow(vertex, tiny, mag, W, H, 0, 0, zoom, SCALE, 0, BIG).len;

    expect(noFloor).toBeLessThan(MIN); // invisibly small without the floor
    expect(withFloor).toBeGreaterThanOrEqual(MIN - 1e-9);
});

test('arrow direction is the true projected direction of the 3D vector', () => {
    const vertex: Vec3 = [0, 0, 0];
    const right = projectArrow(vertex, [1, 0, 0], 1, W, H, 0, 0, zoom, SCALE, 0, BIG);
    expect(right.tipX - right.baseX).toBeGreaterThan(0);
    expect(Math.abs(right.tipY - right.baseY)).toBeLessThan(1e-6);

    const up = projectArrow(vertex, [0, 1, 0], 1, W, H, 0, 0, zoom, SCALE, 0, BIG);
    expect(up.tipY - up.baseY).toBeLessThan(0); // screen y grows downward
    expect(Math.abs(up.tipX - up.baseX)).toBeLessThan(1e-6);
});

test('C: a near-axial gradient does NOT flip direction as its vertex crosses the canvas centre', () => {
    // Reported bug: an arrow "changes direction dramatically near the centre". Cause was the
    // perspective derivative's position-dependent term (X·ds), which for a gradient pointing
    // near the view axis overtook the true in-plane lean and reversed sign across the centre.
    // The in-plane lean here is +X, so the on-screen arrow must point +X REGARDLESS of which
    // side of centre the vertex sits on. @see src/viewRotation.ts (projectArrow, guard 1)
    const negGrad: Vec3 = [0.05, 0, 1]; // tiny +X lean, mostly toward/along the view axis
    const mag = Math.hypot(negGrad[0], negGrad[1], negGrad[2]);

    const rightOfCentre = projectArrow([0.6, 0, 0], negGrad, mag, W, H, 0, 0, zoom, SCALE, 14, 90);
    const leftOfCentre = projectArrow([-0.6, 0, 0], negGrad, mag, W, H, 0, 0, zoom, SCALE, 14, 90);

    expect(rightOfCentre.baseX).toBeGreaterThan(W / 2); // vertex really is right of centre
    expect(leftOfCentre.baseX).toBeLessThan(W / 2); // ...and left of centre
    // Both must point the same way (+X), matching the vector's actual in-plane lean.
    expect(rightOfCentre.tipX - rightOfCentre.baseX).toBeGreaterThan(0);
    expect(leftOfCentre.tipX - leftOfCentre.baseX).toBeGreaterThan(0);
});

test('B: huge (near-intersection) gradients stay bounded & finite — no perspective blow-up', () => {
    const vertex: Vec3 = [0, 0, 0]; // near canvas center
    const MAX = 80;
    // A diagonal direction that (with the old far-endpoint projection) drove the tip
    // toward the perspective near-plane and skewed the arrow off-canvas.
    for (const g of [1e6, 1e9, 1e12, 1e15]) {
        const negGrad: Vec3 = [0.5 * g, 0.3 * g, -0.8 * g];
        const gnorm = Math.hypot(negGrad[0], negGrad[1], negGrad[2]);
        const a = projectArrow(vertex, negGrad, gnorm, W, H, 0.5, 0.3, zoom, SCALE, 14, MAX);
        expect(Number.isFinite(a.tipX)).toBe(true);
        expect(Number.isFinite(a.tipY)).toBe(true);
        expect(a.len).toBeLessThanOrEqual(MAX + 1e-9);
    }
});
