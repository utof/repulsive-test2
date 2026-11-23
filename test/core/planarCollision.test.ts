// Continuous collision test for flat graphs, and the line search guard built on it.
// @see src/core/planarCollision.ts
import { describe, expect, setDefaultTimeout, test } from 'bun:test';
import { type DispatchDescentStepArgs, dispatchDescentStep } from '../../src/core/dispatch';
import { runForceLayout } from '../../src/core/forceLayout';
import { isFlat, planarMotionCollides } from '../../src/core/planarCollision';
import { edgeLengths, totalLength } from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import { countPlanarCrossings } from '../../src/core/subdivide';
import { calculateDisjointPairs } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

setDefaultTimeout(120_000);

// A horizontal bar (0,1) and a separate vertical stick (2,3) to its right / across it.
const BAR_STICK: Edge[] = [
    [0, 1],
    [2, 3],
];
const cfg = (stickX: number, stickLow: number, stickHigh: number): Vec3[] => [
    [0, 0, 0],
    [4, 0, 0],
    [stickX, stickLow, 0],
    [stickX, stickHigh, 0],
];

describe('planarMotionCollides', () => {
    test('a stick lowered through a bar collides; moved beside it does not', () => {
        // Above the bar → across it: the lower end passes through the bar.
        expect(planarMotionCollides(cfg(2, 1, 3), cfg(2, -1, 1), BAR_STICK)).toBe(true);
        // Above the bar → still above it.
        expect(planarMotionCollides(cfg(2, 1, 3), cfg(2, 0.5, 2.5), BAR_STICK)).toBe(false);
        // Past the bar's end, lowered: nothing is passed through.
        expect(planarMotionCollides(cfg(5, 1, 3), cfg(5, -1, 1), BAR_STICK)).toBe(false);
    });

    test('a full pass (start and end both clear of each other) is still a collision', () => {
        // The stick jumps from above the bar to below it: no crossing before or after,
        // which is exactly what comparing end states would miss.
        const from = cfg(2, 1, 3);
        const to = cfg(2, -3, -1);
        expect(countPlanarCrossings(from, BAR_STICK)).toBe(0);
        expect(countPlanarCrossings(to, BAR_STICK)).toBe(0);
        expect(planarMotionCollides(from, to, BAR_STICK)).toBe(true);
    });

    test('a vertex sweeping past the END of an edge is caught through the edge it belongs to', () => {
        // The bar's right end moves up through the stick.
        const from = cfg(5, 1, 3);
        const to: Vec3[] = [
            [0, 0, 0],
            [6, 2, 0],
            [5, 1, 0],
            [5, 3, 0],
        ];
        expect(planarMotionCollides(from, to, BAR_STICK)).toBe(true);
    });

    test('edges that share a vertex are never paired; rotation about it is free', () => {
        const V: Vec3[] = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ];
        const W: Vec3[] = [
            [0, 0, 0],
            [1, 0, 0],
            [0.9, 0.1, 0],
        ];
        const hinge: Edge[] = [
            [0, 1],
            [0, 2],
        ];
        expect(planarMotionCollides(V, W, hinge)).toBe(false);
    });

    test('a candidate that ends under the clearance floor is refused; moving apart is not', () => {
        // Mean edge length 3, floor 3e-6: stopping 1e-9 short of the bar is refused…
        expect(planarMotionCollides(cfg(2, 1, 3), cfg(2, 1e-9, 2), BAR_STICK)).toBe(true);
        // …a pair that STARTS that close may separate…
        expect(planarMotionCollides(cfg(2, 1e-9, 2), cfg(2, 1, 3), BAR_STICK)).toBe(false);
        // …and a comfortable gap is fine.
        expect(planarMotionCollides(cfg(2, 1, 3), cfg(2, 0.01, 2), BAR_STICK)).toBe(false);
    });

    test('off the plane it has no opinion', () => {
        const up = cfg(2, 1, 3).map((p) => [p[0], p[1], 0.5] as Vec3);
        const down = cfg(2, -1, 1).map((p, i) => [p[0], p[1], i === 0 ? 0.6 : 0.5] as Vec3);
        expect(isFlat(cfg(2, 1, 3), cfg(2, -1, 1))).toBe(true);
        expect(isFlat(up, up)).toBe(true); // flat at z = 0.5 counts
        expect(isFlat(up, down)).toBe(false);
        expect(planarMotionCollides(up, down, BAR_STICK)).toBe(false);
    });
});

describe('the crossing guard in a Sobolev step', () => {
    // A binary tree laid out by the force layout from a scramble: MEASURED 3 crossings left.
    let s = 3;
    const r = () => {
        s = (1664525 * s + 1013904223) % 4294967296;
        return s / 4294967296;
    };
    const n = 63;
    const edges: Edge[] = Array.from(
        { length: n - 1 },
        (_, i) => [Math.floor(i / 2), i + 1] as Edge,
    );
    const start = runForceLayout(
        Array.from({ length: n }, () => [4 * r() - 2, 4 * r() - 2, 0] as Vec3),
        edges,
    ).vertices;
    const pairs = calculateDisjointPairs(edges);
    const base = {
        descentMode: 'sobolev',
        vertices: start,
        edges,
        disjointPairs: pairs,
        mode: 'analytical',
        stepSize: 0.001,
        x0: barycenterTarget(start, edges),
        sobolevL0: totalLength(start, edges),
        sobolevEll0: edgeLengths(start, edges),
        barycenterConstraint: true,
        lengthMode: 'total',
        projectionMode: 'frozen',
    } satisfies DispatchDescentStepArgs;
    const flow = (preventCrossings: boolean, steps: number) => {
        let V = start;
        let energy: number | undefined;
        const counts = [countPlanarCrossings(V, edges)];
        let reason: string | undefined;
        for (let k = 0; k < steps; k++) {
            const res = dispatchDescentStep({
                ...base,
                vertices: V,
                preventCrossings,
                ...(energy === undefined ? {} : { energyBefore: energy }),
            });
            if (!res.accepted) {
                reason = res.stats?.reason;
                break;
            }
            V = res.vertices;
            energy = res.energy;
            counts.push(countPlanarCrossings(V, edges));
        }
        return { counts, reason };
    };

    test('without it the flow passes edges through each other', () => {
        const { counts } = flow(false, 30);
        expect(counts[0]).toBe(3);
        // MEASURED: 3 → 0 within 5 steps.
        expect(new Set(counts).size).toBeGreaterThan(1);
    });

    test('with it the crossing count never changes, and a blocked flow says `collision`', () => {
        const { counts, reason } = flow(true, 60);
        expect(counts[0]).toBe(3);
        expect(new Set(counts).size).toBe(1);
        // This start has crossings the energy wants to undo by passing through; the guard
        // refuses, so the run ends on the guard's own reason (MEASURED after ~20 steps),
        // not on a silent pass-through as it did before the clearance floor.
        expect(reason).toBe('collision');
    });

    test('absent or false, the step is the unguarded one, bit for bit', () => {
        const a = dispatchDescentStep(base);
        const b = dispatchDescentStep({ ...base, preventCrossings: false });
        expect(b.vertices).toEqual(a.vertices);
        expect(b.energy).toBe(a.energy);
    });
});
