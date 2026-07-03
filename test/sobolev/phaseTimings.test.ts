import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { sobolevStepSet } from '../../src/core/optimizer';
import {
    barycenterBlock,
    type ConstraintSet,
    totalLength,
    totalLengthBlock,
} from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import { flatten } from '../../src/core/sobolev/layout';
import { calculateDisjointPairs } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';
import { dispatchDescentStep } from '../../src/store';

// Phase-timing collector surface (Task 1). Proves: (a) default-off →
// no `timings` field, (b) opt-in → a schema-shaped ledger whose call counts
// match the step composition (1 gradient assembleA + ≥1 projection iterate,
// each paired 1:1 with a saddle solve), (c) collection NEVER perturbs numerics
// (the whole point of the default-off backstop — the golden suites are the
// wider guard).
// @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 1)

interface Fixture {
    name: string;
    vertices: Vec3[];
    edges: Edge[];
    alpha: number;
    beta: number;
    epsilon: number;
}

function loadFixture(name: string): Fixture {
    return JSON.parse(
        readFileSync(new URL(`../../oracle/fixtures/${name}.json`, import.meta.url), 'utf8'),
    ) as Fixture;
}

function crossingSet(vertices: Vec3[], edges: Edge[]): ConstraintSet {
    // Barycenter FIRST (spec §3.2 row order) + total length: the store's
    // default sobolev constraint set, and the set that yields ≥1 projection
    // iterate on this fixture (crossing-length golden: projection_iterations = 1).
    return [
        barycenterBlock(barycenterTarget(vertices, edges)),
        totalLengthBlock(totalLength(vertices, edges)),
    ];
}

test('phaseTimings: default off — sobolevStepSet result has no `timings` field', () => {
    const { vertices, edges, alpha, beta, epsilon } = loadFixture('crossing');
    const disjointPairs = calculateDisjointPairs(edges);
    const set = crossingSet(vertices, edges);

    const r = sobolevStepSet(vertices, edges, disjointPairs, set, {
        mode: 'analytical',
        alpha,
        beta,
        epsilon,
    });
    expect(r.timings).toBeUndefined();
});

test('phaseTimings: opt-in — schema-shaped ledger, step.calls===1, energy.calls≥2, assembleA.calls≥2, saddle.calls===assembleA.calls', () => {
    const { vertices, edges, alpha, beta, epsilon } = loadFixture('crossing');
    const disjointPairs = calculateDisjointPairs(edges);
    const set = crossingSet(vertices, edges);

    const r = sobolevStepSet(vertices, edges, disjointPairs, set, {
        mode: 'analytical',
        alpha,
        beta,
        epsilon,
        collectTimings: true,
    });
    const t = r.timings;
    expect(t).toBeDefined();
    if (!t) throw new Error('timings missing with collectTimings: true');

    // Every present phase records non-negative wall-clock and ≥1 call.
    for (const sample of Object.values(t)) {
        if (!sample) continue;
        expect(sample.ms).toBeGreaterThanOrEqual(0);
        expect(sample.calls).toBeGreaterThanOrEqual(1);
    }

    expect(t.step?.calls).toBe(1);
    // E₀ + ≥1 Armijo trial.
    expect(t.energy?.calls ?? 0).toBeGreaterThanOrEqual(2);
    // gradient solve + ≥1 projection iterate.
    expect(t.assembleA?.calls ?? 0).toBeGreaterThanOrEqual(2);
    // assembleA and solveSaddle are called 1:1 at both the gradient-solve and
    // every projection-iterate call site.
    expect(t.saddle?.calls).toBe(t.assembleA?.calls);
});

test('phaseTimings: collection does not perturb numerics — vertices/energy/stats identical with vs without the flag', () => {
    const { vertices, edges, alpha, beta, epsilon } = loadFixture('crossing');
    const disjointPairs = calculateDisjointPairs(edges);
    const set = crossingSet(vertices, edges);
    const opts = { mode: 'analytical' as const, alpha, beta, epsilon };

    const off = sobolevStepSet(vertices, edges, disjointPairs, set, opts);
    const on = sobolevStepSet(vertices, edges, disjointPairs, set, {
        ...opts,
        collectTimings: true,
    });

    expect(flatten(on.vertices)).toEqual(flatten(off.vertices));
    expect(on.energy).toBe(off.energy);
    expect(on.stats).toEqual(off.stats);
});

// Task 3 wiring: the store's dispatch passes collectTimings into the sobolev
// step and surfaces `timings` on the outcome; the raw path has no sobolev
// timings (null). @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 3)
test('phaseTimings: dispatchDescentStep — sobolev+flag → non-null timings, raw → null', () => {
    const { vertices, edges } = loadFixture('crossing');
    const disjointPairs = calculateDisjointPairs(edges);
    const x0 = barycenterTarget(vertices, edges);
    const L0 = totalLength(vertices, edges);
    const base = {
        vertices,
        edges,
        disjointPairs,
        mode: 'analytical' as const,
        stepSize: 0.001,
        x0,
    };

    const sob = dispatchDescentStep({
        ...base,
        descentMode: 'sobolev',
        barycenterConstraint: true,
        lengthMode: 'total',
        sobolevL0: L0,
        collectTimings: true,
    });
    expect(sob.timings).not.toBeNull();
    expect(sob.timings?.step?.calls).toBe(1);

    const raw = dispatchDescentStep({ ...base, descentMode: 'raw' });
    expect(raw.timings).toBeNull();
});
