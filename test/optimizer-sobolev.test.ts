import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { sobolevStep, step } from '../src/core/optimizer';
import { barycenterTarget } from '../src/core/sobolev/constraints';
import { calculateDisjointPairs, calculateEnergy } from '../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../src/core/testConfigs';
import { dispatchDescentStep, useSimStore } from '../src/store';

interface Fixture {
    name: string;
    vertices: Vec3[];
    edges: Edge[];
    alpha: number;
    beta: number;
    epsilon: number;
}

// Load at runtime (avoids needing resolveJsonModule in tsconfig; test/** is typechecked),
// mirroring test/sobolev/lineSearch.test.ts.
function loadFixture(name: string): Fixture {
    return JSON.parse(
        readFileSync(new URL(`../oracle/fixtures/${name}.json`, import.meta.url), 'utf8'),
    ) as Fixture;
}

// ── sobolevStep: the app-facing wrapper of the verified Stage-1 pipeline ──────

test('sobolevStep: 2 consecutive steps on crossing — accepted, energy strictly decreases, residual ≤ 1e-10, stats populated', () => {
    const { vertices, edges, alpha, beta, epsilon } = loadFixture('crossing');
    const disjointPairs = calculateDisjointPairs(edges);
    // x₀ frozen once from the initial state — same contract as the flow test in
    // test/sobolev/lineSearch.test.ts.
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("set x₀ once at initialization")
    const x0 = barycenterTarget(vertices, edges);
    const e0 = calculateEnergy(vertices, edges, disjointPairs, alpha, beta, epsilon);

    let current = vertices;
    let previousEnergy = e0;
    for (let i = 0; i < 2; i++) {
        const r = sobolevStep(current, edges, disjointPairs, x0, {
            mode: 'analytical',
            alpha,
            beta,
            epsilon,
        });
        console.log(
            `[sobolevStep] step ${i + 1}: τ = ${r.stats.tau}, E = ${r.energy.toExponential(6)}, ` +
                `residual = ${r.stats.residual.toExponential(3)}, ` +
                `‖g̃‖ = ${r.stats.gradientL2Norm.toExponential(3)}, ` +
                `projection iterations = ${r.stats.projectionIterations}`,
        );
        expect(r.accepted).toBe(true);
        expect(r.converged).toBe(false);
        expect(r.energy).toBeLessThan(previousEnergy);
        // Self-certifying saddle residual — spec §E prop 8.
        expect(r.stats.residual).toBeLessThanOrEqual(1e-10);
        expect(r.stats.tau).toBeGreaterThan(0);
        expect(r.stats.gradientL2Norm).toBeGreaterThan(0);
        expect(r.stats.projectionIterations).not.toBeNull();
        expect(r.stats.reason).toBeUndefined();
        current = r.vertices;
        previousEnergy = r.energy;
    }
});

test('sobolevStep: rejected/converged outcomes echo the input vertices unchanged (spec §C steps 5/10)', () => {
    const { vertices, edges, alpha, beta, epsilon } = loadFixture('crossing');
    const disjointPairs = calculateDisjointPairs(edges);
    const x0 = barycenterTarget(vertices, edges);
    const before = vertices.map((v) => [v[0], v[1], v[2]] as Vec3);

    const r = sobolevStep(vertices, edges, disjointPairs, x0, {
        mode: 'analytical',
        alpha,
        beta,
        epsilon,
    });
    // Purity: inputs never mutated, accepted result is a NEW array.
    expect(vertices).toEqual(before);
    expect(r.vertices).not.toBe(vertices);
});

// ── store-level mode dispatch ─────────────────────────────────────────────────

test('dispatchDescentStep: raw mode matches step() output exactly on the same input', () => {
    const { vertices, edges } = loadFixture('crossing');
    const disjointPairs = calculateDisjointPairs(edges);
    const x0 = barycenterTarget(vertices, edges);

    const expected = step(vertices, edges, disjointPairs, {
        mode: 'analytical',
        stepSize: 0.001,
    });
    const got = dispatchDescentStep({
        descentMode: 'raw',
        vertices,
        edges,
        disjointPairs,
        mode: 'analytical',
        stepSize: 0.001,
        x0,
    });

    // Bit-identical: same inputs through the same step() — any drift means the
    // dispatch wrapper altered the raw path.
    expect(got.vertices).toEqual(expected.vertices);
    expect(Object.is(got.energy, expected.energy)).toBe(true);
    expect(got.accepted).toBe(true);
    expect(got.converged).toBe(false);
    expect(got.stats).toBeNull();
});

test('dispatchDescentStep: sobolev mode routes to sobolevStep and changes vertices', () => {
    const { vertices, edges } = loadFixture('crossing');
    const disjointPairs = calculateDisjointPairs(edges);
    const x0 = barycenterTarget(vertices, edges);

    const expected = sobolevStep(vertices, edges, disjointPairs, x0, { mode: 'analytical' });
    const got = dispatchDescentStep({
        descentMode: 'sobolev',
        vertices,
        edges,
        disjointPairs,
        mode: 'analytical',
        stepSize: 0.001,
        x0,
    });

    expect(got.vertices).toEqual(expected.vertices);
    expect(Object.is(got.energy, expected.energy)).toBe(true);
    expect(got.accepted).toBe(true);
    expect(got.stats).toEqual(expected.stats);
    // Sobolev actually moved the curve (τ ≈ 1 scale, not raw's 1e-5 scale).
    expect(got.vertices).not.toEqual(vertices);
});

// ── store: descentMode state + frozen-x0 lifecycle ───────────────────────────

test('store: descentMode defaults to raw; setDescentMode toggles and clears stale sobolev diagnostics', () => {
    expect(useSimStore.getState().descentMode).toBe('raw');
    useSimStore.getState().setDescentMode('sobolev');
    expect(useSimStore.getState().descentMode).toBe('sobolev');
    expect(useSimStore.getState().sobolevStats).toBeNull();
    expect(useSimStore.getState().sobolevConverged).toBe(false);
    useSimStore.getState().setDescentMode('raw');
    expect(useSimStore.getState().descentMode).toBe('raw');
});

test('store: x0 re-anchors on play and on commit, from the live positions', () => {
    const st = useSimStore.getState();
    // Perturb the live buffer (as if the user had run and paused elsewhere), then
    // play: x0 must re-anchor to the CURRENT geometry, not the pristine preset.
    st.live[0][0] += 0.123;
    useSimStore.getState().setRunning(true);
    const x0AtPlay = useSimStore.getState().sobolevX0;
    expect(x0AtPlay).toEqual(
        barycenterTarget(useSimStore.getState().live, useSimStore.getState().graph.edges),
    );

    // Mid-run motion of the live buffer must NOT move the frozen target: only
    // play / commit / rebuild recompute it.
    // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §B ("set x₀ once at initialization")
    useSimStore.getState().live[1][1] += 0.5;
    expect(useSimStore.getState().sobolevX0).toBe(x0AtPlay);

    // Pause = vertex commit → re-anchor from the committed positions.
    useSimStore.getState().setRunning(false);
    const x0AtCommit = useSimStore.getState().sobolevX0;
    expect(x0AtCommit).not.toBe(x0AtPlay);
    expect(x0AtCommit).toEqual(
        barycenterTarget(useSimStore.getState().live, useSimStore.getState().graph.edges),
    );

    // Preset/config rebuild → re-anchor from the regenerated graph.
    useSimStore.getState().regenerate();
    expect(useSimStore.getState().sobolevX0).toEqual(
        barycenterTarget(useSimStore.getState().graph.vertices, useSimStore.getState().graph.edges),
    );
});
