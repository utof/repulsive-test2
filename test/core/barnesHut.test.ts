/**
 * Barnes–Hut energy and differential against the exact all-pairs reference.
 *
 * What each group can and cannot see:
 *  - "θ = 0" pins the exact part of the traversal (leaf sums, the disjointness rule, both
 *    orderings of the differential) to rounding, on a ring, an open chain and a graph with
 *    junctions. It says nothing about the far field: at θ = 0 no node is ever lumped.
 *  - "far field" pins the approximation on the trefoil, with `farNodes > 0` asserted first so a
 *    traversal that silently fell back to exact sums cannot pass as "accurate".
 *  - "adjacency" is the one guard geometry does not imply; it is shown firing with a knock-out.
 *  - "step seam" runs a descent step that never reads `disjointPairs`.
 *
 * Every bar below is the measured value on this fixture times a stated margin; the measured
 * value is in the comment beside it. They are regression pins for THIS discretisation, not
 * accuracy claims about other curves.
 * @see src/core/barnesHut.ts
 */
import { describe, expect, setDefaultTimeout, test } from 'bun:test';
import {
    type BarnesHutOptions,
    type BarnesHutStats,
    barnesHutEnergy,
    barnesHutGradient,
    prepareBarnesHut,
} from '../../src/core/barnesHut';
import { trefoil } from '../../src/core/fixtures';
import { DEFAULTS, sobolevStepSet } from '../../src/core/optimizer';
import { barycenterBlock, type ConstraintSet } from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import { makeHierarchicalSaddle } from '../../src/core/sobolev/hierarchical/hierarchicalSaddle';
import {
    calculateDisjointPairs,
    calculateEnergy,
    gradientAnalytical,
} from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';
import { lazy } from '../tier';

// Why: the n = 960 far-field references are lazy getters (test/tier.ts `lazy`), so the first test
// of that describe pays for them (~2.7 s isolated, measured 5.1 s at load average 18 in the
// four-process runner), which bun's 5 s default per-test timeout does not cover.
setDefaultTimeout(60_000);

const { alpha: ALPHA, beta: BETA, epsilon: EPS } = DEFAULTS;

type Curve = { vertices: Vec3[]; edges: Edge[] };

/** Open helix: n vertices, n − 1 edges, two free ends. */
function openHelix(n: number): Curve {
    const vertices: Vec3[] = [];
    const edges: Edge[] = [];
    for (let i = 0; i < n; i++) {
        const t = (6 * Math.PI * i) / (n - 1);
        vertices.push([Math.cos(t), Math.sin(t), 0.15 * t]);
        if (i > 0) edges.push([i - 1, i]);
    }
    return { vertices, edges };
}

/** Three arcs joining two poles: two vertices of degree 3, so "adjacent" is not "index ± 1". */
function thetaGraph(perArc: number): Curve {
    const vertices: Vec3[] = [
        [0, 0, 1],
        [0, 0, -1],
    ];
    const edges: Edge[] = [];
    for (let a = 0; a < 3; a++) {
        const phi = (2 * Math.PI * a) / 3 + 0.3;
        let prev = 0;
        for (let i = 1; i <= perArc; i++) {
            const t = (Math.PI * i) / (perArc + 1);
            const r = Math.sin(t) * (1 + 0.2 * a);
            vertices.push([r * Math.cos(phi), r * Math.sin(phi), Math.cos(t)]);
            edges.push([prev, vertices.length - 1]);
            prev = vertices.length - 1;
        }
        edges.push([prev, 1]);
    }
    return { vertices, edges };
}

/**
 * A right-angle path I–J–K of three unit edges, plus a tiny 40-edge ring around J's midpoint.
 * The ring pulls J into a BVH node whose midpoints all lie within ~0.02 of each other, so the
 * node is small and, by the spatial test alone, far from I and from K — yet it contains J, which
 * shares a vertex with both. Only the adjacency check keeps J out of their far field. The corner
 * matters: with I, J, K collinear the wrongly lumped term is `|T_I × d| = 0` and nothing shows.
 */
function cornerWithRing(): Curve {
    const vertices: Vec3[] = [
        [-1, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
    ];
    const edges: Edge[] = [
        [0, 1],
        [1, 2],
        [2, 3],
    ];
    const ring = 40;
    for (let i = 0; i < ring; i++) {
        const t = (2 * Math.PI * i) / ring;
        vertices.push([0.004 * Math.cos(t), 0.5 + 0.004 * Math.sin(t), 0.02]);
        edges.push([4 + i, 4 + ((i + 1) % ring)]);
    }
    return { vertices, edges };
}

function exact(c: Curve) {
    const dp = calculateDisjointPairs(c.edges);
    return {
        dp,
        E: calculateEnergy(c.vertices, c.edges, dp, ALPHA, BETA, EPS),
        g: gradientAnalytical(c.vertices, c.edges, dp, ALPHA, BETA, EPS).flat(),
    };
}

function relErr(a: number[], b: number[]): number {
    let d = 0;
    let nb = 0;
    for (let i = 0; i < a.length; i++) {
        d += (a[i] - b[i]) ** 2;
        nb += b[i] ** 2;
    }
    return Math.sqrt(d / nb);
}

function bh(c: Curve, opts: BarnesHutOptions) {
    const ctx = prepareBarnesHut(c.vertices, c.edges, EPS, opts);
    const eStats: BarnesHutStats = { farNodes: 0, nearPairs: 0 };
    const gStats: BarnesHutStats = { farNodes: 0, nearPairs: 0 };
    const E = barnesHutEnergy(ctx, ALPHA, BETA, eStats);
    const g = barnesHutGradient(ctx, ALPHA, BETA, gStats).flat();
    return { E, g, eStats, gStats };
}

const log = (label: string, value: unknown) => console.log(`[barnesHut] ${label}: ${value}`);

describe('θ = 0 reproduces the all-pairs energy and differential to rounding', () => {
    const cases: [string, Curve][] = [
        ['trefoil ring, n = 64', trefoil(64)],
        ['open helix, n = 50', openHelix(50)],
        ['theta graph, two degree-3 junctions', thetaGraph(20)],
        ['right-angle path with a ring at the middle edge', cornerWithRing()],
    ];
    for (const [name, c] of cases) {
        for (const traversal of ['vertex', 'edge'] as const) {
            test(`${name} — ${traversal} traversal`, () => {
                const ref = exact(c);
                const r = bh(c, { theta: 0, traversal });
                // No node may be lumped at θ = 0 — otherwise agreement would be an accident.
                expect(r.eStats.farNodes).toBe(0);
                expect(r.gStats.farNodes).toBe(0);
                // Measured ≤ 3e-14 on every case; the bar leaves two decades for summation order.
                expect(Math.abs(r.E - ref.E) / ref.E).toBeLessThan(1e-11);
                expect(relErr(r.g, ref.g)).toBeLessThan(1e-11);
            });
        }
    }
});

describe('far field on the trefoil, n = 960', () => {
    // Lazy getters, not describe-body work (test/tier.ts `lazy`). Why: the exact all-pairs
    // reference and four Barnes–Hut evaluations at n = 960 (~2.7 s) ran at collection, so every
    // `-t` filtered run of this file paid them. Shared by the tests below, read-only.
    const c = trefoil(960);
    const ref = lazy(() => exact(c));
    const at = (theta: number, traversal: 'vertex' | 'edge' = 'vertex') => {
        const r = bh(c, { theta, traversal });
        return {
            ...r,
            eErr: Math.abs(r.E - ref().E) / ref().E,
            gErr: relErr(r.g, ref().g),
        };
    };
    const t010 = lazy(() => at(0.1));
    const t025 = lazy(() => at(0.25));
    const t050 = lazy(() => at(0.5));
    const edge025 = lazy(() => at(0.25, 'edge'));

    test('the default θ = 0.25 lumps nodes, and is accurate', () => {
        log('n=960 θ=0.25 energy relErr', t025().eErr);
        log('n=960 θ=0.25 gradient relErr (vertex)', t025().gErr);
        log(
            'n=960 θ=0.25 energy far/near',
            `${t025().eStats.farNodes} / ${t025().eStats.nearPairs}`,
        );
        expect(t025().eStats.farNodes).toBeGreaterThan(0);
        expect(t025().gStats.farNodes).toBeGreaterThan(0);
        // The exact sum is 960·957 = 918 720 ordered pairs; the far field must have replaced most.
        expect(t025().eStats.nearPairs).toBeLessThan(918720 / 4);
        // MEASURED 3.29e-3 (energy) and 1.45e-2 (differential); bars are 2× those.
        expect(t025().eErr).toBeLessThan(6.6e-3);
        expect(t025().gErr).toBeLessThan(2.9e-2);
    });

    test('the error falls as θ falls', () => {
        log(
            'n=960 gradient relErr θ=0.1 / 0.25 / 0.5',
            `${t010().gErr} / ${t025().gErr} / ${t050().gErr}`,
        );
        log(
            'n=960 energy relErr θ=0.1 / 0.25 / 0.5',
            `${t010().eErr} / ${t025().eErr} / ${t050().eErr}`,
        );
        // MEASURED differential 8.85e-4 / 1.45e-2 / 1.16e-1: each step is at least 5×.
        expect(t010().gErr * 5).toBeLessThan(t025().gErr);
        expect(t025().gErr * 5).toBeLessThan(t050().gErr);
        // Energy: MEASURED 1.54e-5 at θ = 0.1 against 3.29e-3 at 0.25. θ = 0.5 is NOT asserted
        // above 0.25: its signed error happens to be the smaller in magnitude here (−2.98e-3).
        expect(t010().eErr * 5).toBeLessThan(t025().eErr);
    });

    test('per-vertex traversal beats the per-edge traversal of the paper', () => {
        log('n=960 θ=0.25 gradient relErr (edge)', edge025().gErr);
        // MEASURED 1.45e-2 (vertex) against 4.97e-2 (edge), a factor 3.4. This is the reason for
        // the default; see barnesHutGradient's TSDoc for the mechanism.
        expect(t025().gErr * 2).toBeLessThan(edge025().gErr);
    });
});

describe('graphs with junctions and free ends keep their accuracy', () => {
    test('theta graph, 3 × 100 interior vertices, θ = 0.25', () => {
        const c = thetaGraph(100);
        const ref = exact(c);
        const r = bh(c, { theta: 0.25 });
        const eErr = Math.abs(r.E - ref.E) / ref.E;
        const gErr = relErr(r.g, ref.g);
        log('theta graph θ=0.25 energy / gradient relErr', `${eErr} / ${gErr}`);
        log(
            'theta graph θ=0.25 far nodes (energy / gradient)',
            `${r.eStats.farNodes} / ${r.gStats.farNodes}`,
        );
        expect(r.eStats.farNodes).toBeGreaterThan(0);
        expect(r.gStats.farNodes).toBeGreaterThan(0);
        // MEASURED 1.26e-3 (energy) and 8.5e-5 (differential); bars are 2× those. The
        // differential is far more accurate here than on the trefoil because this graph is not
        // near equilibrium: its gradient is large and involves little cancellation.
        expect(eErr).toBeLessThan(2.6e-3);
        expect(gErr).toBeLessThan(1.8e-4);
    });

    test('open helix, n = 400, θ = 0.25', () => {
        const c = openHelix(400);
        const ref = exact(c);
        const r = bh(c, { theta: 0.25 });
        const eErr = Math.abs(r.E - ref.E) / ref.E;
        const gErr = relErr(r.g, ref.g);
        log('open helix θ=0.25 energy / gradient relErr', `${eErr} / ${gErr}`);
        log(
            'open helix θ=0.25 far nodes (energy / gradient)',
            `${r.eStats.farNodes} / ${r.gStats.farNodes}`,
        );
        expect(r.eStats.farNodes).toBeGreaterThan(0);
        expect(r.gStats.farNodes).toBeGreaterThan(0);
        // MEASURED 1.99e-3 (energy) and 2.64e-3 (differential); bars are 2× those.
        expect(eErr).toBeLessThan(4e-3);
        expect(gErr).toBeLessThan(5.3e-3);
    });
});

describe('adjacency: a node holding a neighbouring edge is never lumped', () => {
    const c = cornerWithRing();
    const ref = exact(c);
    // `thetaT: 10` switches the tangential test off so that the SPATIAL test is the only thing
    // standing between the ring-plus-J node and the differential's far field; the ring's tangents
    // point every way, and the default thetaT would otherwise hide the defect for the gradient.
    const ok = bh(c, { theta: 0.25, thetaT: 10 });
    const ko = bh(c, { theta: 0.25, thetaT: 10, knockOut: 'no-adjacency-check' });
    const eErr = (r: typeof ok) => Math.abs(r.E - ref.E) / ref.E;

    test('the knock-out lumps nodes the guard refuses', () => {
        log(
            'adjacency far nodes with check / knocked out',
            `${ok.eStats.farNodes} / ${ko.eStats.farNodes}`,
        );
        // MEASURED 2 against 4: the two extra are the node holding J, seen from I and from K.
        expect(ok.eStats.farNodes).toBeGreaterThan(0);
        expect(ko.eStats.farNodes).toBeGreaterThan(ok.eStats.farNodes);
    });

    test('with the check both quantities are accurate; knocked out, neither is', () => {
        log('adjacency energy relErr with check / knocked out', `${eErr(ok)} / ${eErr(ko)}`);
        log(
            'adjacency gradient relErr with check / knocked out',
            `${relErr(ok.g, ref.g)} / ${relErr(ko.g, ref.g)}`,
        );
        // MEASURED energy 6.8e-10 against 6.0e-3, differential 1.0e-7 against 9.9e-5. The bars
        // sit between the two with a decade to spare on each side.
        expect(eErr(ok)).toBeLessThan(1e-8);
        expect(eErr(ko)).toBeGreaterThan(1e-3);
        expect(relErr(ok.g, ref.g)).toBeLessThan(1e-6);
        expect(relErr(ko.g, ref.g)).toBeGreaterThan(1e-5);
    });
});

describe('step seam: a descent step that never reads disjointPairs', () => {
    const n = 240;
    const c = trefoil(n);
    const set: ConstraintSet = [barycenterBlock(barycenterTarget(c.vertices, c.edges))];

    test('hierarchical solve + Barnes–Hut energy and differential, disjointPairs = []', () => {
        const ref = exact(c);
        const energyOf = (V: Vec3[]) =>
            barnesHutEnergy(prepareBarnesHut(V, c.edges, EPS, { theta: 0.25 }), ALPHA, BETA);
        const dE = barnesHutGradient(
            prepareBarnesHut(c.vertices, c.edges, EPS, { theta: 0.25 }),
            ALPHA,
            BETA,
        );
        const saddle = makeHierarchicalSaddle(c.vertices, c.edges, null);
        const out = sobolevStepSet(c.vertices, c.edges, [], set, {
            mode: 'analytical',
            projectionMode: 'frozen',
            saddle,
            dE,
            tpeEnergy: energyOf,
        });
        // The step's own energy is the Barnes–Hut one; judge it on the EXACT energy instead.
        const after = calculateEnergy(out.vertices, c.edges, ref.dp, ALPHA, BETA, EPS);
        log('n=240 step exact energy before / after', `${ref.E} / ${after}`);
        expect(out.accepted).toBe(true);
        expect(after).toBeLessThan(ref.E);
    });

    test('makeHierarchicalSaddle still rejects a pair list built for another curve', () => {
        const wrong = calculateDisjointPairs(trefoil(n - 1).edges);
        expect(() => makeHierarchicalSaddle(c.vertices, c.edges, wrong)).toThrow(RangeError);
    });
});
