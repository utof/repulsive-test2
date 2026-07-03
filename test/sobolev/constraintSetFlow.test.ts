import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { sobolevStep, sobolevStepSet } from '../../src/core/optimizer';
import {
    barycenterBlock,
    type ConstraintSet,
    evaluateConstraintSet,
    totalLength,
    totalLengthBlock,
} from '../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../src/core/sobolev/constraints';
import { solveConstrainedGradientSet } from '../../src/core/sobolev/gradient';
import { flatten } from '../../src/core/sobolev/layout';
import { lineSearchStepSet } from '../../src/core/sobolev/lineSearch';
import { calculateDisjointPairs, calculateEnergy } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

// The two constraints-M1 golden pairs (crossing + one loop fixture), generated
// by oracle/tpe_constraints_oracle.py — never the five stage-1 goldens.
// @see oracle/README.md ("Constraints goldens")
// @see docs/superpowers/specs/2026-07-03-sobolev-constraints-design.md §4.4
const FIXTURE_NAMES = ['crossing', 'linked-rings'] as const;

// Mirrored line-search tunable, used only to re-verify Armijo from RETURNED
// numbers. OUR constant, not the paper's.
// @see local_files/2026-07-02-sobolev-formula-audit.md (Item 9)
const ARMIJO_C1 = 1e-4;

interface Fixture {
    name: string;
    vertices: Vec3[];
    edges: Edge[];
    alpha: number;
    beta: number;
    epsilon: number;
}

interface GoldenLength {
    dE: Vec3[];
    x0_barycenter_target: Vec3;
    L0_total_length_target: number;
    g_tilde: Vec3[];
    g_tilde_flat: number[];
    lambda: number[];
    gradient_l2_norm: number;
    line_search_step: {
        accepted: boolean;
        tau: number;
        energy_before: number;
        energy_after: number;
        slope: number;
        gradient_l2_norm: number;
        vertices: Vec3[];
        projection_iterations: number;
        projection_phi_norm: number;
    };
}

// Load at runtime (avoids needing resolveJsonModule in tsconfig; test/** is
// typechecked), mirroring test/sobolev/gradient.test.ts.
function loadFixture(name: string): Fixture {
    return JSON.parse(
        readFileSync(new URL(`../../oracle/fixtures/${name}.json`, import.meta.url), 'utf8'),
    ) as Fixture;
}

function loadGolden(name: string): GoldenLength {
    return JSON.parse(
        readFileSync(new URL(`../../oracle/golden/${name}-length.json`, import.meta.url), 'utf8'),
    ) as GoldenLength;
}

function euclideanNorm(a: number[]): number {
    let sumSq = 0;
    for (const x of a) sumSq += x * x;
    return Math.sqrt(sumSq);
}

function euclideanDiff(a: number[], b: number[]): number {
    let sumSq = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        sumSq += d * d;
    }
    return Math.sqrt(sumSq);
}

function matVec(M: number[][], v: number[]): number[] {
    return M.map((row) => row.reduce((s, x, j) => s + x * v[j], 0));
}

function dot(a: number[], b: number[]): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
}

function lengthConstrainedSet(golden: GoldenLength): ConstraintSet {
    // Barycenter FIRST — the row-order rule of spec §3.2.
    return [
        barycenterBlock(golden.x0_barycenter_target),
        totalLengthBlock(golden.L0_total_length_target),
    ];
}

for (const name of FIXTURE_NAMES) {
    const fixture = loadFixture(name);
    const golden = loadGolden(name);

    // DESIGN DECISION (preserve): inputs are the oracle's own outputs
    // (golden.dE, golden targets), NOT TS-side recomputations — decouples the
    // solve comparison from cross-language FD noise. Same rationale as
    // test/sobolev/gradient.test.ts / lineSearch.test.ts.
    // @see oracle/README.md ("Known tolerance caveats")
    test(`solveConstrainedGradientSet: ${name} — matches oracle g̃/λ to 1e-9, residual ≤ 1e-10, descent, C·g̃ ≈ 0`, () => {
        const { vertices, edges, alpha, beta, epsilon } = fixture;
        const disjointPairs = calculateDisjointPairs(edges);
        const set = lengthConstrainedSet(golden);

        const { gTilde, lambda, residual } = solveConstrainedGradientSet(
            vertices,
            edges,
            disjointPairs,
            alpha,
            beta,
            epsilon,
            golden.dE,
            set,
        );
        const gFlat = flatten(gTilde);

        const gRelDiff =
            euclideanDiff(gFlat, golden.g_tilde_flat) / euclideanNorm(golden.g_tilde_flat);
        const lambdaRelDiff =
            euclideanDiff(lambda, golden.lambda) / Math.max(1, euclideanNorm(golden.lambda));
        const descentDot = dot(flatten(golden.dE), gFlat);
        // Constraint check recomputes the stacked C independently of the solve.
        const { C } = evaluateConstraintSet(set, vertices, edges);
        const constraintRel = euclideanNorm(matVec(C, gFlat)) / Math.max(1, euclideanNorm(gFlat));

        console.log(
            `[gradientSet] ${name} (|V| = ${vertices.length}): g̃ rel diff = ${gRelDiff.toExponential(3)}, ` +
                `λ rel diff = ${lambdaRelDiff.toExponential(3)} (k = ${lambda.length}), ` +
                `residual = ${residual.toExponential(3)}, ‖C·g̃‖/max(1,‖g̃‖) = ${constraintRel.toExponential(3)}`,
        );

        // 4 constraint rows: 3 barycenter + 1 total length (spec §2).
        expect(lambda.length).toBe(4);
        expect(gRelDiff).toBeLessThanOrEqual(1e-9);
        expect(lambdaRelDiff).toBeLessThanOrEqual(1e-9);
        // Self-certifying saddle residual — spec §E prop 8.
        expect(residual).toBeLessThanOrEqual(1e-10);
        expect(descentDot).toBeGreaterThan(0);
        expect(constraintRel).toBeLessThanOrEqual(1e-10);
    });

    test(`lineSearchStepSet: ${name} — matches oracle acceptance, τ, iterations, energy, vertices`, () => {
        const { vertices, edges, alpha, beta, epsilon } = fixture;
        const disjointPairs = calculateDisjointPairs(edges);
        const gold = golden.line_search_step;
        const set = lengthConstrainedSet(golden);

        const result = lineSearchStepSet(
            vertices,
            edges,
            disjointPairs,
            alpha,
            beta,
            epsilon,
            golden.dE,
            golden.g_tilde,
            set,
        );

        const energyRelDiff =
            Math.abs(result.energyAfter - gold.energy_after) / Math.abs(gold.energy_after);
        const vertexRelDiff =
            euclideanDiff(flatten(result.vertices), flatten(gold.vertices)) /
            euclideanNorm(flatten(gold.vertices));

        console.log(
            `[lineSearchSet] ${name}: accepted = ${result.accepted}, τ = ${result.tau}, ` +
                `projection iterations = ${result.projectionIterations}, ` +
                `E ${result.energyBefore.toExponential(6)} → ${result.energyAfter.toExponential(6)} ` +
                `(rel diff vs oracle = ${energyRelDiff.toExponential(3)}), ` +
                `vertices rel diff = ${vertexRelDiff.toExponential(3)}`,
        );

        expect(gold.accepted).toBe(true);
        expect(result.accepted).toBe(gold.accepted);
        // τ EXACT (powers of two — bit-identical across languages; drift means
        // the accept/reject LOGIC diverged). Same gate as lineSearch.test.ts.
        expect(result.tau).toBe(gold.tau);
        expect(result.projectionIterations).toBe(gold.projection_iterations);
        expect(energyRelDiff).toBeLessThanOrEqual(1e-12);
        expect(vertexRelDiff).toBeLessThanOrEqual(1e-9);

        // Armijo re-verified from RETURNED numbers — spec §C step 9.
        expect(result.slope).toBeDefined();
        expect(result.energyAfter).toBeLessThanOrEqual(
            result.energyBefore - ARMIJO_C1 * result.tau * (result.slope as number),
        );
    });
}

// Back-compat bit-identity (spec §4.4.4 + §3.2): the x0 wrapper and the
// explicit barycenter-only set must produce IDENTICAL outputs — this is the
// regression proof that the ConstraintSet refactor did not perturb the
// pre-existing barycenter-only path.
test('back-compat: sobolevStep(x0) ≡ sobolevStepSet([barycenterBlock(x0)]) bit-identically on crossing', () => {
    const { vertices, edges, alpha, beta, epsilon } = loadFixture('crossing');
    const disjointPairs = calculateDisjointPairs(edges);
    const x0 = barycenterTarget(vertices, edges);
    const opts = { mode: 'analytical' as const, alpha, beta, epsilon };

    const viaX0 = sobolevStep(vertices, edges, disjointPairs, x0, opts);
    const viaSet = sobolevStepSet(vertices, edges, disjointPairs, [barycenterBlock(x0)], opts);

    expect(viaX0.accepted).toBe(true);
    expect(viaSet.energy).toBe(viaX0.energy);
    expect(viaSet.accepted).toBe(viaX0.accepted);
    expect(viaSet.converged).toBe(viaX0.converged);
    expect(flatten(viaSet.vertices)).toEqual(flatten(viaX0.vertices));
    expect(viaSet.stats).toEqual(viaX0.stats);
});

// Flow property (spec §4.4.3, oracle-independent): ≥5 sobolev steps on
// crossing with barycenter + total length — every step accepted, energy
// strictly decreases, and after EVERY step the length drift satisfies the
// projection stopping rule |L − L⁰| ≤ 1e-4·max(1, L). The bound MIRRORS the
// rule the default projection enforces (value 1e-4 = reference-impl
// backproj_threshold — see oracle/README.md "Projection tolerance
// provenance"); it is per-step tolerance, not accumulated drift.
test('flow: 5 steps on crossing with barycenter+length — accepted, energy ↓, |L−L⁰| within stopping rule each step', () => {
    const { vertices, edges, alpha, beta, epsilon } = loadFixture('crossing');
    const disjointPairs = calculateDisjointPairs(edges);
    // Frozen targets: computed ONCE from the initial state (x₀/L⁰ lifecycle,
    // spec §3.5) — never re-anchored during the run.
    const x0 = barycenterTarget(vertices, edges);
    const L0 = totalLength(vertices, edges);
    const set: ConstraintSet = [barycenterBlock(x0), totalLengthBlock(L0)];
    const opts = { mode: 'analytical' as const, alpha, beta, epsilon };

    let current = vertices;
    let previousEnergy = calculateEnergy(current, edges, disjointPairs, alpha, beta, epsilon);

    for (let step = 0; step < 5; step++) {
        const r = sobolevStepSet(current, edges, disjointPairs, set, opts);
        expect(r.accepted).toBe(true);
        expect(r.energy).toBeLessThan(previousEnergy);

        current = r.vertices;
        previousEnergy = r.energy;

        const L1 = totalLength(current, edges);
        const driftBound = 1e-4 * Math.max(1, L1);
        console.log(
            `[flowSet] step ${step + 1}: τ = ${r.stats.tau}, E = ${r.energy.toExponential(6)}, ` +
                `‖g̃‖ = ${r.stats.gradientL2Norm.toExponential(3)}, |L−L⁰| = ${Math.abs(L1 - L0).toExponential(3)} ` +
                `(bound ${driftBound.toExponential(3)})`,
        );
        expect(Math.abs(L1 - L0)).toBeLessThanOrEqual(driftBound);
    }
});

// Empty-set support (spec §9a, user-requested all-constraint toggles): k = 0
// constraint rows — the saddle system degenerates to Ā·g̃ = dE, projection is
// trivially converged, and the unconstrained Sobolev step still descends.
test('empty ConstraintSet: sobolevStepSet takes an accepted, energy-decreasing step on crossing', () => {
    const { vertices, edges, alpha, beta, epsilon } = loadFixture('crossing');
    const disjointPairs = calculateDisjointPairs(edges);
    const e0 = calculateEnergy(vertices, edges, disjointPairs, alpha, beta, epsilon);

    const r = sobolevStepSet(vertices, edges, disjointPairs, [], {
        mode: 'analytical',
        alpha,
        beta,
        epsilon,
    });

    expect(r.accepted).toBe(true);
    expect(r.energy).toBeLessThan(e0);
    // Trivial projection: converged at iteration 0 with no constraint rows.
    expect(r.stats.projectionIterations).toBe(0);
    expect(r.stats.residual).toBeLessThanOrEqual(1e-10);
});
