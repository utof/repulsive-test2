import { expect, test } from 'bun:test';
import { sobolevStep } from '../src/core/optimizer';
import { barycenterTarget } from '../src/core/sobolev/constraints';
import { calculateDisjointPairs } from '../src/core/tangentPointEnergy';
import { testConfigs } from '../src/core/testConfigs';

// An isolated vertex (in no edge) has all-zero rows in the Sobolev saddle
// system (nothing assembles into its Ā rows or C columns) → the solve is
// exactly singular and sobolev mode can never step. Verified empirically:
// 40 random 20v/30e graphs, singular ⟺ isolated ≥ 1 with no exceptions.
// @see src/core/sobolev/linsolve.ts (luSolve singularity throw)
test('random graph preset: every vertex is in at least one edge (singular-Ā guard)', () => {
    const random = testConfigs.find((t) => t.id === 'random');
    expect(random).toBeDefined();
    for (let trial = 0; trial < 20; trial++) {
        const { vertices, edges } = random!.generate({ vertices: 20, edges: 30 });
        const used = new Set(edges.flat());
        expect(used.size).toBe(vertices.length);
        // and edges stay valid indices
        for (const [a, b] of edges) {
            expect(a).toBeGreaterThanOrEqual(0);
            expect(b).toBeGreaterThanOrEqual(0);
            expect(a).toBeLessThan(vertices.length);
            expect(b).toBeLessThan(vertices.length);
            expect(a).not.toBe(b);
        }
    }
});

// End-to-end guard for the reported app crash: a sobolev step on a freshly
// generated random graph must return (accepted or rejected) — NEVER throw.
// @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §C (step 10 — reject, report, don't die)
test('random graph preset: sobolevStep never throws', () => {
    const random = testConfigs.find((t) => t.id === 'random');
    for (let trial = 0; trial < 5; trial++) {
        const { vertices, edges } = random!.generate({ vertices: 20, edges: 30 });
        const disjointPairs = calculateDisjointPairs(edges);
        const x0 = barycenterTarget(vertices, edges);
        const r = sobolevStep(vertices, edges, disjointPairs, x0, { mode: 'analytical' });
        expect(typeof r.accepted).toBe('boolean');
        expect(r.stats).toBeDefined();
    }
});
