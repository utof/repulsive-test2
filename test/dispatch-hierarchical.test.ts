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

    test('fixed total length', () => {
        const hier = dispatchDescentStep({ ...base, sobolevSolver: 'hierarchical' });
        expect(dense.accepted).toBe(true);
        expect(hier.accepted).toBe(true);
        // Same line-search outcome, and the same exact energy. MEASURED 40.5363 (dense) against
        // 40.5359 (hierarchical) from 41.2401: the two decreases agree to 6e-4 relative.
        expect(hier.stats?.tau).toBe(dense.stats?.tau);
        const dDense = e0 - exact(dense.vertices);
        const dHier = e0 - exact(hier.vertices);
        expect(dDense).toBeGreaterThan(0);
        expect(Math.abs(dHier - dDense) / dDense).toBeLessThan(5e-3);
        // The constraint holds: MEASURED relative length drift 2.6e-5 after the step.
        const L = totalLength(hier.vertices, g.edges);
        expect(Math.abs(L - base.sobolevL0) / base.sobolevL0).toBeLessThan(1e-4);
    });

    test('with a pinned vertex', () => {
        const pins = [{ vertexIndex: 10, target: [...g.vertices[10]] as Vec3, enabled: true }];
        const d = dispatchDescentStep({ ...base, pins, sobolevSolver: 'dense' });
        const h = dispatchDescentStep({ ...base, pins, sobolevSolver: 'hierarchical' });
        expect(h.accepted).toBe(true);
        const dD = e0 - exact(d.vertices);
        const dH = e0 - exact(h.vertices);
        // MEASURED 40.6102 (dense) against 40.6191 (hierarchical): 1.4e-2 relative on the decrease.
        expect(Math.abs(dH - dD) / dD).toBeLessThan(5e-2);
        // The pin is held to the projection's tolerance, not exactly: MEASURED 2.0e-5.
        const p = h.vertices[10];
        const t = pins[0].target;
        expect(Math.hypot(p[0] - t[0], p[1] - t[1], p[2] - t[2])).toBeLessThan(1e-4);
    });

    test('with every edge length fixed', () => {
        const d = dispatchDescentStep({ ...base, lengthMode: 'perEdge', sobolevSolver: 'dense' });
        const h = dispatchDescentStep({
            ...base,
            lengthMode: 'perEdge',
            sobolevSolver: 'hierarchical',
        });
        expect(h.accepted).toBe(true);
        const dD = e0 - exact(d.vertices);
        const dH = e0 - exact(h.vertices);
        // MEASURED 40.6840 (dense) against 40.6850 (hierarchical): 1.8e-3 relative.
        expect(Math.abs(dH - dD) / dD).toBeLessThan(1e-2);
    });

    test('no `sobolevSolver` is the dense route, bit for bit', () => {
        const absent = dispatchDescentStep(base);
        expect(absent.energy).toBe(dense.energy);
        expect(absent.vertices).toEqual(dense.vertices);
    });

    test('the hierarchical step reports its tree and build times under dE and assembleA', () => {
        const h = dispatchDescentStep({
            ...base,
            sobolevSolver: 'hierarchical',
            collectTimings: true,
        });
        expect(h.timings?.dE?.ms).toBeGreaterThan(0);
        expect(h.timings?.assembleA?.ms).toBeGreaterThan(0);
        // No dense factorization ran on this route.
        expect(h.timings?.factor).toBeUndefined();
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
