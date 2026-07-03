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
import { calculateDisjointPairs, calculateEnergy } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

// E₀ reuse across steps (Task 4). Proves: passing `energyBefore` — when it is
// EXACTLY calculateEnergy(current vertices, …), as a continuous run's previous
// accepted step guarantees — is bit-identical to recomputing E₀ inside the
// step, while dropping one calculateEnergy call (the E₀ eval) per step.
// @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 4)

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
    // default sobolev constraint set — same set the phaseTimings suite uses to
    // exercise an accepted line-search step (E₀ + ≥1 Armijo eval).
    return [
        barycenterBlock(barycenterTarget(vertices, edges)),
        totalLengthBlock(totalLength(vertices, edges)),
    ];
}

for (const name of ['crossing', 'linked-rings']) {
    test(`energyReuse: ${name} — energyBefore is bit-identical to recomputed E₀`, () => {
        const { vertices, edges, alpha, beta, epsilon } = loadFixture(name);
        const disjointPairs = calculateDisjointPairs(edges);
        const set = crossingSet(vertices, edges);
        const opts = { mode: 'analytical' as const, alpha, beta, epsilon };

        // e0 = calculateEnergy(current vertices) — exactly the value the previous
        // accepted step would have returned at these vertices (the invariant that
        // makes the reuse legal). @see plan Task 4 correctness invariant.
        const e0 = calculateEnergy(vertices, edges, disjointPairs, alpha, beta, epsilon);

        const off = sobolevStepSet(vertices, edges, disjointPairs, set, opts);
        const on = sobolevStepSet(vertices, edges, disjointPairs, set, {
            ...opts,
            energyBefore: e0,
        });

        expect(flatten(on.vertices)).toEqual(flatten(off.vertices));
        expect(on.energy).toBe(off.energy);
        expect(on.stats).toEqual(off.stats);
    });

    test(`energyReuse: ${name} — reuse drops exactly one energy eval`, () => {
        const { vertices, edges, alpha, beta, epsilon } = loadFixture(name);
        const disjointPairs = calculateDisjointPairs(edges);
        const set = crossingSet(vertices, edges);
        const opts = { mode: 'analytical' as const, alpha, beta, epsilon, collectTimings: true };
        const e0 = calculateEnergy(vertices, edges, disjointPairs, alpha, beta, epsilon);

        const off = sobolevStepSet(vertices, edges, disjointPairs, set, opts);
        const on = sobolevStepSet(vertices, edges, disjointPairs, set, {
            ...opts,
            energyBefore: e0,
        });

        // With reuse the E₀ calculateEnergy call is skipped, so the 'energy'
        // phase records exactly one fewer call (?? 0 handles a converged/singular
        // step where the reused path fires zero energy calls, dropping the key).
        expect(on.timings?.energy?.calls ?? 0).toBe((off.timings?.energy?.calls ?? 0) - 1);
    });
}
