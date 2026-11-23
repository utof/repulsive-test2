/**
 * `dispatchDescentStep` on the hierarchical route — the function the app's frame loop and its
 * worker both call. Every energy asserted here is the EXACT all-pairs energy re-evaluated on the
 * step's output, never the Barnes–Hut energy the step itself reports: the step's own number
 * cannot be the evidence that the step went downhill.
 *
 * The bars are measured values on these fixtures with the stated margins.
 * @see src/core/dispatch.ts, src/core/sobolevRoute.ts
 */
import { describe, expect, setDefaultTimeout, test } from 'bun:test';
import { barnesHutEnergy, prepareBarnesHut } from '../src/core/barnesHut';
import { type DispatchDescentStepArgs, dispatchDescentStep } from '../src/core/dispatch';
import { DEFAULTS } from '../src/core/optimizer';
import { edgeLengths, totalLength } from '../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../src/core/sobolev/constraints';
import { PAIR_LIST_MAX_N } from '../src/core/sobolevRoute';
import { calculateDisjointPairs, calculateEnergy } from '../src/core/tangentPointEnergy';
import { type Edge, testConfigs, type Vec3 } from '../src/core/testConfigs';
import { buildGraphState } from '../src/store';

setDefaultTimeout(120_000);

const { alpha: ALPHA, beta: BETA, epsilon: EPS } = DEFAULTS;
const largeKnot = testConfigs.find((t) => t.id === 'knot-large');
if (!largeKnot) throw new Error('the knot-large preset is missing');

function argsFor(vertices: Vec3[], edges: Edge[], disjointPairs: number[][]) {
    return {
        descentMode: 'sobolev',
        vertices,
        edges,
        disjointPairs,
        mode: 'analytical',
        stepSize: 0.001,
        x0: barycenterTarget(vertices, edges),
        sobolevL0: totalLength(vertices, edges),
        sobolevEll0: edgeLengths(vertices, edges),
        barycenterConstraint: true,
        lengthMode: 'total',
        projectionMode: 'frozen',
    } satisfies DispatchDescentStepArgs;
}

describe('n = 480 trefoil: the two routes take the same step', () => {
    const g = largeKnot.generate({ vertices: 480 });
    const dp = calculateDisjointPairs(g.edges);
    const exact = (V: Vec3[]) => calculateEnergy(V, g.edges, dp, ALPHA, BETA, EPS);
    const base = argsFor(g.vertices, g.edges, dp);
    const e0 = exact(g.vertices);
    const dense = dispatchDescentStep({ ...base, sobolevSolver: 'dense' });

    // Both linear solvers of the hierarchical route, forced, against the dense route. The bars
    // are per solver: tree + GMRES at tolerance 1e-2 agrees to a few 1e-3; the direct
    // factorization solves exactly and, on this fixture, also sums the differential exactly
    // (the tree does not engage: 21% of pairs are near-field), so it reproduces the dense step.
    // MEASURED relative difference of the exact-energy decrease, gmres / direct:
    //   fixed length 1.7e-3 / 1.7e-12, pinned vertex 4.9e-3 / 9.0e-14, per-edge 5.4e-4 / 6.7e-12.
    const SOLVERS = [
        { solver: 'gmres', total: 5e-3, pin: 5e-2, perEdge: 1e-2, pinHeld: 1e-4 },
        { solver: 'direct', total: 1e-9, pin: 1e-9, perEdge: 1e-9, pinHeld: 1e-12 },
    ] as const;
    for (const bar of SOLVERS) {
        const hierarchical = (extra: Partial<DispatchDescentStepArgs> = {}) =>
            dispatchDescentStep({
                ...base,
                ...extra,
                sobolevSolver: 'hierarchical',
                hierarchicalLinearSolver: bar.solver,
            });

        test(`${bar.solver}: fixed total length`, () => {
            const hier = hierarchical();
            expect(dense.accepted).toBe(true);
            expect(hier.accepted).toBe(true);
            // Same line-search outcome, and the same exact energy (41.2401 → 40.536).
            expect(hier.stats?.tau).toBe(dense.stats?.tau);
            const dDense = e0 - exact(dense.vertices);
            const dHier = e0 - exact(hier.vertices);
            expect(dDense).toBeGreaterThan(0);
            expect(Math.abs(dHier - dDense) / dDense).toBeLessThan(bar.total);
            // The constraint holds: MEASURED relative length drift 2.5e-5 after the step.
            const L = totalLength(hier.vertices, g.edges);
            expect(Math.abs(L - base.sobolevL0) / base.sobolevL0).toBeLessThan(1e-4);
        });

        test(`${bar.solver}: with a pinned vertex`, () => {
            const pins = [{ vertexIndex: 10, target: [...g.vertices[10]] as Vec3, enabled: true }];
            const d = dispatchDescentStep({ ...base, pins, sobolevSolver: 'dense' });
            const h = hierarchical({ pins });
            expect(h.accepted).toBe(true);
            const dD = e0 - exact(d.vertices);
            const dH = e0 - exact(h.vertices);
            expect(Math.abs(dH - dD) / dD).toBeLessThan(bar.pin);
            // The pin: MEASURED 1.9e-5 (gmres, the projection's tolerance) / 6.0e-15 (direct).
            const p = h.vertices[10];
            const t = pins[0].target;
            expect(Math.hypot(p[0] - t[0], p[1] - t[1], p[2] - t[2])).toBeLessThan(bar.pinHeld);
        });

        test(`${bar.solver}: with every edge length fixed`, () => {
            const d = dispatchDescentStep({
                ...base,
                lengthMode: 'perEdge',
                sobolevSolver: 'dense',
            });
            const h = hierarchical({ lengthMode: 'perEdge' });
            expect(h.accepted).toBe(true);
            const dD = e0 - exact(d.vertices);
            const dH = e0 - exact(h.vertices);
            expect(Math.abs(dH - dD) / dD).toBeLessThan(bar.perEdge);
        });
    }

    test('no `sobolevSolver` is the dense route, bit for bit', () => {
        const absent = dispatchDescentStep(base);
        expect(absent.energy).toBe(dense.energy);
        expect(absent.vertices).toEqual(dense.vertices);
    });

    test('the step reports its build under dE and assembleA, and `factor` only when direct', () => {
        const run = (solver: 'gmres' | 'direct' | 'auto') =>
            dispatchDescentStep({
                ...base,
                sobolevSolver: 'hierarchical',
                hierarchicalLinearSolver: solver,
                collectTimings: true,
            }).timings;
        const tree = run('gmres');
        expect(tree?.dE?.ms).toBeGreaterThan(0);
        expect(tree?.assembleA?.ms).toBeGreaterThan(0);
        expect(tree?.factor).toBeUndefined(); // no factorization on the tree route
        expect(run('direct')?.factor?.calls).toBe(1);
        // 'auto' on this fixture is the direct route: 21% of its pairs are near-field, against
        // a threshold of 9.5% at n = 480 (directIsCheaper).
        expect(run('auto')?.factor?.calls).toBe(1);
    });
});

describe(`above PAIR_LIST_MAX_N (${PAIR_LIST_MAX_N}): no pair list anywhere`, () => {
    const n = PAIR_LIST_MAX_N + 500;
    const built = buildGraphState(largeKnot, { vertices: n });
    // The exact energy WITHOUT a pair list: Barnes–Hut at θ = 0 lumps nothing.
    const exact = (V: Vec3[]) =>
        barnesHutEnergy(prepareBarnesHut(V, built.graph.edges, EPS, { theta: 0 }), ALPHA, BETA);

    test('the store builds no pair list and shows a Barnes–Hut energy', () => {
        expect(built.graph.vertices.length).toBe(n);
        expect(built.disjointPairs).toEqual([]);
        const e = exact(built.graph.vertices);
        // MEASURED offset of the Barnes–Hut energy at θ = 0.25: +0.3–0.5 %. Bar: 1 %.
        expect(Math.abs(built.energy - e) / e).toBeLessThan(1e-2);
    });

    test('a Sobolev step runs, is accepted, and lowers the exact energy', () => {
        const base = argsFor(built.graph.vertices, built.graph.edges, built.disjointPairs);
        const r = dispatchDescentStep({ ...base, sobolevSolver: 'auto' });
        expect(r.accepted).toBe(true);
        expect(exact(r.vertices)).toBeLessThan(exact(built.graph.vertices));
    });

    test('raw descent runs on the Barnes–Hut differential', () => {
        const base = argsFor(built.graph.vertices, built.graph.edges, built.disjointPairs);
        const r = dispatchDescentStep({ ...base, descentMode: 'raw', stepSize: 1e-9 });
        expect(r.accepted).toBe(true);
        expect(Number.isFinite(r.energy)).toBe(true);
        expect(r.vertices).not.toEqual(built.graph.vertices);
    });

    test('an open chain of that size is refused, not crashed', () => {
        const vertices = built.graph.vertices;
        const edges = built.graph.edges.slice(0, n - 1);
        const base = argsFor(vertices, edges, []);
        const r = dispatchDescentStep({ ...base, sobolevSolver: 'auto', energyBefore: 1 });
        expect(r.accepted).toBe(false);
        expect(r.vertices).toBe(vertices);
        expect(r.energy).toBe(1);
    });
});

describe('a whole flow on the hierarchical route ends on a verdict, not on a stall', () => {
    // A 5-turn coil around a ring: an unknot that relaxes to a round circle of the same length.
    const n = 300;
    const vertices: Vec3[] = [];
    const edges: Edge[] = [];
    for (let i = 0; i < n; i++) {
        const t = (2 * Math.PI * i) / n;
        const r = 1 + 0.33 * Math.cos(5 * t);
        vertices.push([r * Math.cos(t), r * Math.sin(t), 0.33 * Math.sin(5 * t)]);
        edges.push([i, (i + 1) % n]);
    }
    const dp = calculateDisjointPairs(edges);
    const exact = (V: Vec3[]) => calculateEnergy(V, edges, dp, ALPHA, BETA, EPS);
    const coarseBh = (V: Vec3[]) =>
        barnesHutEnergy(prepareBarnesHut(V, edges, EPS, { theta: 0.25 }), ALPHA, BETA);

    test('it converges, uses the refined setting on the way, and keeps the energy contract', () => {
        const base = argsFor(vertices, edges, dp);
        let V = vertices;
        let energy: number | undefined;
        let refinedSteps = 0;
        let steps = 0;
        let last = dispatchDescentStep({ ...base, sobolevSolver: 'hierarchical' });
        for (; steps < 80; steps++) {
            last = dispatchDescentStep({
                ...base,
                vertices: V,
                sobolevSolver: 'hierarchical',
                collectTimings: true,
                ...(energy === undefined ? {} : { energyBefore: energy }),
            });
            if (last.timings?.dE?.calls === 2) refinedSteps++;
            if (!last.accepted) break;
            // The contract the frame loop relies on: the returned energy is the COARSE
            // Barnes–Hut energy of the returned vertices, whichever attempt produced them —
            // bit for bit, since it is the same deterministic evaluation.
            expect(last.energy).toBe(coarseBh(last.vertices));
            V = last.vertices;
            energy = last.energy;
        }
        // MEASURED: converged after 35 steps, coarse energy 53.275 -> 1.704.
        expect(last.converged).toBe(true);
        expect(steps).toBeLessThan(60);
        expect(refinedSteps).toBeGreaterThan(0);
        const e = exact(V);
        expect(e).toBeLessThan(exact(vertices) / 20);
        // The dense route agrees that this is the minimum: it converges from here too.
        const dense = dispatchDescentStep({ ...base, vertices: V, sobolevSolver: 'dense' });
        expect(dense.converged).toBe(true);
    });
});
