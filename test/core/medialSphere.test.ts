// The medial-sphere flow and its independent knot-type audit.
// @see docs/plans/2026-11-24-medial-sphere-method.md
import { describe, expect, test } from 'bun:test';
import { knotSignature } from '../../src/core/knotDeterminant';
import {
    type ClosedCurve,
    curveLength,
    medialSphereScales,
    medialSphereStep,
    medialSpheres,
    motionIsSafe,
    resampleClosedCurve,
} from '../../src/core/medialSphere';

function sample(n: number, f: (t: number) => [number, number, number]): ClosedCurve {
    const x = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) x.set(f((2 * Math.PI * i) / n), 3 * i);
    return { n, x };
}
const circle = (n: number, rho = 1) => sample(n, (t) => [rho * Math.cos(t), rho * Math.sin(t), 0]);
const trefoil = (n: number) =>
    sample(n, (t) => [
        Math.sin(t) + 2 * Math.sin(2 * t),
        Math.cos(t) - 2 * Math.cos(2 * t),
        -Math.sin(3 * t),
    ]);
const figureEight = (n: number) =>
    sample(n, (t) => [
        (2 + Math.cos(2 * t)) * Math.cos(3 * t),
        (2 + Math.cos(2 * t)) * Math.sin(3 * t),
        Math.sin(4 * t),
    ]);
/** A deterministic smooth wobble, to make a tidy curve untidy without changing its type. */
function wobble(c: ClosedCurve, amp: number): ClosedCurve {
    const x = Float64Array.from(c.x);
    for (let i = 0; i < c.n; i++) {
        const t = (2 * Math.PI * i) / c.n;
        x[3 * i] += amp * Math.sin(7 * t + 1);
        x[3 * i + 1] += amp * Math.cos(5 * t + 2);
        x[3 * i + 2] += amp * Math.sin(9 * t + 3);
    }
    return { n: c.n, x };
}

describe('knotSignature — the audit', () => {
    test('circle 1, trefoil 3, figure-eight 5', () => {
        expect(knotSignature(circle(60))).toEqual([1, 1]);
        expect(knotSignature(trefoil(120))).toEqual([3, 3]);
        expect(knotSignature(figureEight(160))).toEqual([5, 5]);
    });
    test('unchanged by a wobble that adds crossings to the picture', () => {
        expect(knotSignature(wobble(trefoil(400), 0.25))).toEqual([3, 3]);
        expect(knotSignature(wobble(circle(400, 2), 0.3))).toEqual([1, 1]);
    });
});

describe('medialSpheres', () => {
    test('on a circle the inward sphere is the circle itself, the outward one is capped', () => {
        const rho = 2;
        const N = 16;
        const s = medialSpheres(circle(200, rho), 3 * rho, N);
        for (const i of [0, 37, 111]) {
            let min = Infinity;
            let capped = 0;
            for (let l = 0; l < N; l++) {
                min = Math.min(min, s.radius[i * N + l]);
                if (s.hit[i * N + l] < 0) capped++;
            }
            // The sampled direction nearest to "inward" is at most π/16 off it: R = ρ / cos φ.
            expect(min).toBeGreaterThan(rho * 0.999);
            expect(min).toBeLessThan((rho / Math.cos(Math.PI / N)) * 1.001);
            // The outward half of the normal circle sees no node at all.
            expect(capped).toBeGreaterThanOrEqual(N / 2 - 1);
        }
    });
});

describe('resampleClosedCurve', () => {
    test('spacing ends in [h/2, 2h] and the shape is kept', () => {
        const c = circle(500, 1);
        const h = 0.1;
        const r = resampleClosedCurve(c, h);
        expect(r.n).toBeLessThan(c.n);
        for (let i = 0; i < r.n; i++) {
            const j = (i + 1) % r.n;
            const d = Math.hypot(
                r.x[3 * j] - r.x[3 * i],
                r.x[3 * j + 1] - r.x[3 * i + 1],
                r.x[3 * j + 2] - r.x[3 * i + 2],
            );
            expect(d).toBeLessThanOrEqual(2 * h + 1e-12);
            expect(d).toBeGreaterThan(h / 2);
            expect(Math.hypot(r.x[3 * i], r.x[3 * i + 1])).toBeCloseTo(1, 12);
        }
    });
    test('a node is not dropped when another strand pierces its corner triangle', () => {
        // A hairpin corner at node 1 (short edges, short chord) with a strand through it.
        const pts = [
            [-0.05, 0, 0],
            [0, 0.08, 0],
            [0.05, 0, 0], // the corner: nodes 0,1,2
            [1, 0, 0],
            [1, 0, 1],
            [0, 0.04, 1], // away and over
            [0, 0.04, 0.02],
            [0, 0.04, -0.02], // the piercing strand: nodes 6→7
            [0, 0.04, -1],
            [-1, 0, -1],
            [-1, 0, 0], // and back
        ];
        const c: ClosedCurve = { n: pts.length, x: Float64Array.from(pts.flat()) };
        const r = resampleClosedCurve(c, 0.1);
        // The corner's apex must survive (its y is the only 0.08 in the curve).
        let apex = false;
        for (let i = 0; i < r.n; i++) if (r.x[3 * i + 1] === 0.08) apex = true;
        expect(apex).toBe(true);
        expect(knotSignature(r)).toEqual(knotSignature(c));
    });
});

describe('motionIsSafe', () => {
    // Two far-apart loops' worth of geometry in one closed curve: a long thin rectangle whose
    // two long sides are distance 1 apart.
    const rect = (): ClosedCurve => {
        const pts: number[] = [];
        for (let i = 0; i <= 10; i++) pts.push(i, 0, 0);
        for (let i = 10; i >= 0; i--) pts.push(i, 1, 0);
        return { n: pts.length / 3, x: Float64Array.from(pts) };
    };
    test('a motion that carries one side through the other is refused', () => {
        const c = rect();
        const d = new Float64Array(3 * c.n);
        for (let i = 0; i <= 10; i++) d[3 * i + 1] = -2; // x − step·d: the y=0 side moves up by 2
        expect(motionIsSafe(c, d, 1)).toBe(false);
        expect(motionIsSafe(c, d, 0.2)).toBe(true);
    });
    test('a rigid translation of everything is always safe', () => {
        const c = rect();
        const d = new Float64Array(3 * c.n);
        for (let i = 0; i < c.n; i++) d[3 * i + 1] = -5;
        expect(motionIsSafe(c, d, 1)).toBe(true);
    });
});

describe('medialSphereStep — the flow', () => {
    function run(c0: ClosedCurve, steps: number) {
        const scales = medialSphereScales(c0, 1.5);
        let c = resampleClosedCurve(c0, scales.h);
        let tryStep: number | undefined;
        for (let k = 0; k < steps && c.n > 10; k++) {
            const r = medialSphereStep(c, scales, { tryStep });
            tryStep = r.nextTryStep;
            c = r.curve;
        }
        return c;
    }
    test('a wobbly trefoil tightens and stays a trefoil', () => {
        const c0 = wobble(trefoil(600), 0.25);
        const c = run(c0, 300);
        expect(curveLength(c)).toBeLessThan(0.8 * curveLength(c0));
        expect(c.n).toBeGreaterThan(10);
        expect(knotSignature(c)).toEqual([3, 3]);
    });
    test('a wobbly unknot shrinks away (10 nodes or fewer) and stays an unknot on the way', () => {
        const c0 = wobble(circle(600, 2), 0.3);
        const scales = medialSphereScales(c0, 1.5);
        let c = resampleClosedCurve(c0, scales.h);
        let tryStep: number | undefined;
        for (let k = 0; k < 600 && c.n > 10; k++) {
            const r = medialSphereStep(c, scales, { tryStep });
            tryStep = r.nextTryStep;
            c = r.curve;
            if (k % 20 === 0) expect(knotSignature(c)).toEqual([1, 1]);
        }
        expect(c.n).toBeLessThanOrEqual(10);
    });
});
