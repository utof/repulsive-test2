/**
 * Penalties (5C) — unit + golden tests for src/core/sobolev/penalties.ts.
 *
 * Gates (plan §5): central-FD check of every analytic penalty gradient at
 * rel ≤ 1e-6; TS analytic gradient/energy vs the oracle's recorded values at
 * 1e-12 (both sides are the SAME mirrored formulas — no FD noise in this
 * comparison); V_int exclusion semantics; field-penalty limit cases;
 * degenerate-edge zero contribution.
 * @see docs/superpowers/plans/2026-07-03-sobolev-penalties.md §2, §4 Task 3
 * @see oracle/tpe_constraints_oracle.py (penalty_energy / penalty_gradient / fd_penalty_check)
 */
import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { sobolevStepSet } from '../../src/core/optimizer';
import { barycenterBlock, totalLengthBlock } from '../../src/core/sobolev/constraintSet';
import { solveConstrainedGradientSet } from '../../src/core/sobolev/gradient';
import { flatten } from '../../src/core/sobolev/layout';
import { lineSearchStepSet } from '../../src/core/sobolev/lineSearch';
import {
    type PenaltyConfig,
    penaltiesActive,
    penaltyEnergy,
    penaltyGradient,
} from '../../src/core/sobolev/penalties';
import { calculateDisjointPairs, calculateEnergy } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

interface Fixture {
    vertices: Vec3[];
    edges: Edge[];
}

interface PenaltyGolden {
    alpha: number;
    beta: number;
    epsilon: number;
    penalties: {
        preset: string;
        w_length: number;
        w_diff: number;
        w_field: number;
        X: [number, number, number];
    };
    penalty_energy_initial: number;
    objective_energy_initial: number;
    dE_penalty_flat: number[];
    dE: Vec3[];
    g_tilde: Vec3[];
    g_tilde_flat: number[];
    x0_barycenter_target: Vec3;
    L0_total_length_target?: number;
    line_search_step: {
        accepted: boolean;
        tau: number;
        energy_before: number;
        energy_after: number;
        vertices: Vec3[];
        projection_iterations: number;
    };
}

// Barycenter first (spec §3.2); the pen-combo golden runs the hard
// totalLength constraint alongside the penalties (composition case).
function setFromGolden(gold: PenaltyGolden) {
    return gold.L0_total_length_target !== undefined
        ? [
              barycenterBlock(gold.x0_barycenter_target),
              totalLengthBlock(gold.L0_total_length_target),
          ]
        : [barycenterBlock(gold.x0_barycenter_target)];
}

function loadFixture(name: string): Fixture {
    return JSON.parse(
        readFileSync(new URL(`../../oracle/fixtures/${name}.json`, import.meta.url), 'utf8'),
    ) as Fixture;
}

function loadGolden(file: string): PenaltyGolden {
    return JSON.parse(
        readFileSync(new URL(`../../oracle/golden/${file}.json`, import.meta.url), 'utf8'),
    ) as PenaltyGolden;
}

// Golden `penalties` payload → TS config (weights 0 mean "absent").
function configFromGolden(g: PenaltyGolden): PenaltyConfig {
    const p = g.penalties;
    return {
        ...(p.w_length !== 0 ? { totalLength: p.w_length } : {}),
        ...(p.w_diff !== 0 ? { lengthDiff: p.w_diff } : {}),
        ...(p.w_field !== 0 ? { field: { weight: p.w_field, X: p.X } } : {}),
    };
}

function euclideanNorm(a: number[]): number {
    let s = 0;
    for (const x of a) s += x * x;
    return Math.sqrt(s);
}

function euclideanDiff(a: number[], b: number[]): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        s += d * d;
    }
    return Math.sqrt(s);
}

// Central-difference FD check along the oracle's deterministic direction
// h_i = sin(1 + 0.7 i), normalized — mirror of fd_penalty_check (no RNG).
// @see oracle/tpe_constraints_oracle.py (fd_penalty_check)
function fdCheck(vertices: Vec3[], edges: Edge[], config: PenaltyConfig): number {
    const n = vertices.length;
    const h: number[] = Array.from({ length: 3 * n }, (_, i) => Math.sin(1 + 0.7 * i));
    const hNorm = euclideanNorm(h);
    for (let i = 0; i < h.length; i++) h[i] /= hNorm;
    // Coordinate-block layout [x0..xN-1, y0..yN-1, z0..zN-1] — same as flatten.
    const perturb = (sign: number, eta: number): Vec3[] =>
        vertices.map((v, i) => [
            v[0] + sign * eta * h[i],
            v[1] + sign * eta * h[n + i],
            v[2] + sign * eta * h[2 * n + i],
        ]);
    const eta = 1e-6;
    const ep = penaltyEnergy(perturb(1, eta), edges, config);
    const em = penaltyEnergy(perturb(-1, eta), edges, config);
    const fd = (ep - em) / (2 * eta);
    const gFlat = flatten(penaltyGradient(vertices, edges, config));
    let gh = 0;
    for (let i = 0; i < gFlat.length; i++) gh += gFlat[i] * h[i];
    return Math.abs(fd - gh) / Math.max(1, Math.abs(gh));
}

// --- FD gradient gates on the golden fixtures (plan §5: rel ≤ 1e-6) --------

const FD_CASES: Array<{ fixture: string; config: PenaltyConfig; label: string }> = [
    { fixture: 'crossing', config: { totalLength: 0.5 }, label: 'totalLength' },
    { fixture: 'junction-y', config: { lengthDiff: 10 }, label: 'lengthDiff' },
    { fixture: 'helix', config: { field: { weight: 0.5, X: [1, 0, 1] } }, label: 'field' },
    {
        fixture: 'crossing',
        config: { totalLength: 0.25, lengthDiff: 5, field: { weight: 0.25, X: [1, 0, 1] } },
        label: 'combo',
    },
];

for (const { fixture, config, label } of FD_CASES) {
    test(`penaltyGradient FD check (${label} on ${fixture}): rel ≤ 1e-6`, () => {
        const { vertices, edges } = loadFixture(fixture);
        const rel = fdCheck(vertices, edges, config);
        console.log(`[penalties] FD ${label} on ${fixture}: rel = ${rel.toExponential(3)}`);
        expect(rel).toBeLessThanOrEqual(1e-6);
    });
}

// --- analytic-vs-oracle: SAME formulas both sides, so 1e-12 not 1e-6 -------

const GOLDEN_CASES: Array<{ fixture: string; golden: string }> = [
    { fixture: 'crossing', golden: 'crossing-bary-pen-length' },
    { fixture: 'junction-y', golden: 'junction-y-bary-pen-diff' },
    { fixture: 'helix', golden: 'helix-bary-pen-field' },
    { fixture: 'crossing', golden: 'crossing-length-pen-combo' },
];

for (const { fixture, golden } of GOLDEN_CASES) {
    test(`penaltyEnergy/penaltyGradient vs oracle golden ${golden}: rel ≤ 1e-12`, () => {
        const { vertices, edges } = loadFixture(fixture);
        const gold = loadGolden(golden);
        const config = configFromGolden(gold);

        const e = penaltyEnergy(vertices, edges, config);
        const eRel =
            Math.abs(e - gold.penalty_energy_initial) /
            Math.max(1, Math.abs(gold.penalty_energy_initial));
        const gFlat = flatten(penaltyGradient(vertices, edges, config));
        const gRel =
            euclideanDiff(gFlat, gold.dE_penalty_flat) /
            Math.max(1, euclideanNorm(gold.dE_penalty_flat));

        console.log(
            `[penalties] ${golden}: E_pen = ${e.toExponential(6)} (rel ${eRel.toExponential(3)}), ` +
                `grad rel = ${gRel.toExponential(3)}`,
        );
        expect(eRel).toBeLessThanOrEqual(1e-12);
        expect(gRel).toBeLessThanOrEqual(1e-12);
    });
}

// --- V_int semantics (paper line 764: degree EXACTLY 2) ---------------------

test('lengthDiff: star junction (no degree-2 vertex) contributes nothing', () => {
    // Degree-3 center, degree-1 leaves — V_int is empty even though the edge
    // lengths differ (1, 2, 3).
    const vertices: Vec3[] = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3],
    ];
    const edges: Edge[] = [
        [0, 1],
        [0, 2],
        [0, 3],
    ];
    const config: PenaltyConfig = { lengthDiff: 10 };
    expect(penaltyEnergy(vertices, edges, config)).toBe(0);
    expect(flatten(penaltyGradient(vertices, edges, config)).every((x) => x === 0)).toBe(true);
});

test('lengthDiff: open 3-vertex path — only the interior vertex contributes', () => {
    // Lengths 1 and 2 ⇒ E = w·(1−2)² = w; endpoints are degree-1 (excluded),
    // but they still RECEIVE gradient through the interior vertex's stencil.
    const vertices: Vec3[] = [
        [0, 0, 0],
        [1, 0, 0],
        [3, 0, 0],
    ];
    const edges: Edge[] = [
        [0, 1],
        [1, 2],
    ];
    const w = 7;
    const config: PenaltyConfig = { lengthDiff: w };
    expect(penaltyEnergy(vertices, edges, config)).toBeCloseTo(w, 12);
    // d = ℓ_I − ℓ_J = −1, both tangents +x̂:
    // edge I=(0,1): g[0] −= 2w·d·T = +2w·x̂ → g[0].x = +2w; g[1] += 2w·d·T → −2w
    // edge J=(1,2): g[1] += 2w·d·T ... sign per plan §2.2: J gets the OPPOSITE
    // pattern: g[1] += −(−2w) ... verified numerically by the FD gate; here we
    // just pin the exact stencil values.
    const g = penaltyGradient(vertices, edges, config);
    expect(g[0][0]).toBeCloseTo(2 * w, 12);
    expect(g[1][0]).toBeCloseTo(-4 * w, 12);
    expect(g[2][0]).toBeCloseTo(2 * w, 12);
    // No off-axis leakage.
    expect(g[0][1]).toBe(0);
    expect(g[0][2]).toBe(0);
});

// --- field limit cases (plan §2.3) ------------------------------------------

test('field: edge parallel to X is stationary (E = 0, gradient = 0)', () => {
    const vertices: Vec3[] = [
        [0, 0, 0],
        [2, 0, 0],
    ];
    const edges: Edge[] = [[0, 1]];
    const config: PenaltyConfig = { field: { weight: 0.5, X: [1, 0, 0] } };
    expect(penaltyEnergy(vertices, edges, config)).toBeCloseTo(0, 14);
    const g = flatten(penaltyGradient(vertices, edges, config));
    for (const x of g) expect(Math.abs(x)).toBeLessThanOrEqual(1e-14);
});

test('field: edge perpendicular to X — E = w·ℓ, gradient = ±w·T', () => {
    const w = 0.5;
    const ell = 3;
    const vertices: Vec3[] = [
        [0, 0, 0],
        [0, ell, 0],
    ];
    const edges: Edge[] = [[0, 1]];
    const config: PenaltyConfig = { field: { weight: w, X: [1, 0, 0] } };
    expect(penaltyEnergy(vertices, edges, config)).toBeCloseTo(w * ell, 12);
    const g = penaltyGradient(vertices, edges, config);
    // u = T·X = 0 ⇒ g_I = T (plan §2.3 perpendicular limit).
    expect(g[0][1]).toBeCloseTo(-w, 12);
    expect(g[1][1]).toBeCloseTo(w, 12);
});

// --- degenerate edge + activity ---------------------------------------------

test('degenerate edge (ℓ < 1e-14) contributes zero to every penalty', () => {
    const vertices: Vec3[] = [
        [1, 1, 1],
        [1, 1, 1],
    ];
    const edges: Edge[] = [[0, 1]];
    const config: PenaltyConfig = {
        totalLength: 1,
        lengthDiff: 1,
        field: { weight: 1, X: [0, 0, 1] },
    };
    expect(penaltyEnergy(vertices, edges, config)).toBe(0);
    expect(flatten(penaltyGradient(vertices, edges, config)).every((x) => x === 0)).toBe(true);
});

// --- threading (5C Task 4): line search / solve / full step vs goldens ------

for (const { fixture, golden } of GOLDEN_CASES) {
    test(`lineSearchStepSet with penalties vs oracle ${golden}: τ exact, objective 1e-12, vertices 1e-9`, () => {
        const { vertices, edges } = loadFixture(fixture);
        const gold = loadGolden(golden);
        const config = configFromGolden(gold);
        const disjointPairs = calculateDisjointPairs(edges);
        const set = setFromGolden(gold);

        // Oracle's own dE/g̃ as inputs (decoupled from cross-language FD
        // noise — the constraintSetFlow.test.ts design decision).
        const result = lineSearchStepSet(
            vertices,
            edges,
            disjointPairs,
            gold.alpha,
            gold.beta,
            gold.epsilon,
            gold.dE,
            gold.g_tilde,
            set,
            { penalties: config },
        );
        const goldStep = gold.line_search_step;
        const e0Rel =
            Math.abs(result.energyBefore - goldStep.energy_before) /
            Math.abs(goldStep.energy_before);
        const e1Rel =
            Math.abs(result.energyAfter - goldStep.energy_after) / Math.abs(goldStep.energy_after);
        const vRel =
            euclideanDiff(flatten(result.vertices), flatten(goldStep.vertices)) /
            euclideanNorm(flatten(goldStep.vertices));

        console.log(
            `[penalties] lineSearch ${golden}: τ = ${result.tau}, iters = ${result.projectionIterations}, ` +
                `E₀ rel ${e0Rel.toExponential(3)}, E₁ rel ${e1Rel.toExponential(3)}, vertices rel ${vRel.toExponential(3)}`,
        );
        expect(goldStep.accepted).toBe(true);
        expect(result.accepted).toBe(true);
        // τ and iteration counts EXACT — divergence means accept/reject logic
        // differs, not rounding (constraintSetFlow.test.ts gate).
        expect(result.tau).toBe(goldStep.tau);
        expect(result.projectionIterations).toBe(goldStep.projection_iterations);
        expect(e0Rel).toBeLessThanOrEqual(1e-12);
        expect(e1Rel).toBeLessThanOrEqual(1e-12);
        expect(vRel).toBeLessThanOrEqual(1e-9);
    });
}

test('solveConstrainedGradientSet on the bary set with penalty dE_total matches oracle g̃ (helix)', () => {
    const { vertices, edges } = loadFixture('helix');
    const gold = loadGolden('helix-bary-pen-field');
    const disjointPairs = calculateDisjointPairs(edges);
    const { gTilde, residual } = solveConstrainedGradientSet(
        vertices,
        edges,
        disjointPairs,
        gold.alpha,
        gold.beta,
        gold.epsilon,
        gold.dE,
        setFromGolden(gold),
    );
    const gRel =
        euclideanDiff(flatten(gTilde), gold.g_tilde_flat) / euclideanNorm(gold.g_tilde_flat);
    console.log(
        `[penalties] solve helix-bary: g̃ rel = ${gRel.toExponential(3)}, residual = ${residual.toExponential(3)}`,
    );
    expect(gRel).toBeLessThanOrEqual(1e-9);
    expect(residual).toBeLessThanOrEqual(1e-10);
});

test('sobolevStepSet: absent / empty / all-zero penalty configs are bit-identical', () => {
    const { vertices, edges } = loadFixture('crossing');
    const gold = loadGolden('crossing-bary-pen-length');
    const disjointPairs = calculateDisjointPairs(edges);
    const set = [barycenterBlock(gold.x0_barycenter_target)];
    const base = {
        mode: 'analytical' as const,
        alpha: gold.alpha,
        beta: gold.beta,
        epsilon: gold.epsilon,
    };

    const plain = sobolevStepSet(vertices, edges, disjointPairs, set, base);
    const empty = sobolevStepSet(vertices, edges, disjointPairs, set, { ...base, penalties: {} });
    const zeros = sobolevStepSet(vertices, edges, disjointPairs, set, {
        ...base,
        penalties: { totalLength: 0, lengthDiff: 0, field: { weight: 0.5, X: [0, 0, 0] } },
    });

    // Deep numeric equality — the plan §2.4 bit-identity gate (same pattern
    // as frozenProjection.test.ts's default-path guard).
    expect(empty).toEqual(plain);
    expect(zeros).toEqual(plain);
});

test('sobolevStepSet with penalties: returned energy IS the total objective at the returned vertices', () => {
    const { vertices, edges } = loadFixture('crossing');
    const gold = loadGolden('crossing-bary-pen-length');
    const config = configFromGolden(gold);
    const disjointPairs = calculateDisjointPairs(edges);
    const set = [barycenterBlock(gold.x0_barycenter_target)];

    const result = sobolevStepSet(vertices, edges, disjointPairs, set, {
        mode: 'analytical',
        alpha: gold.alpha,
        beta: gold.beta,
        epsilon: gold.epsilon,
        penalties: config,
    });
    expect(result.accepted).toBe(true);
    const expected =
        calculateEnergy(
            result.vertices,
            edges,
            disjointPairs,
            gold.alpha,
            gold.beta,
            gold.epsilon,
        ) + penaltyEnergy(result.vertices, edges, config);
    // Exact: the step computed its energy through the identical expression
    // (plan §2.4 — E₀-reuse chaining depends on this).
    expect(result.energy).toBe(expected);
});

test('penaltiesActive: zero/absent/degenerate-X configs are inactive', () => {
    expect(penaltiesActive(undefined)).toBe(false);
    expect(penaltiesActive({})).toBe(false);
    expect(penaltiesActive({ totalLength: 0, lengthDiff: 0 })).toBe(false);
    expect(penaltiesActive({ field: { weight: 0.5, X: [0, 0, 0] } })).toBe(false);
    expect(penaltiesActive({ totalLength: 0.5 })).toBe(true);
    expect(penaltiesActive({ lengthDiff: 2 })).toBe(true);
    expect(penaltiesActive({ field: { weight: 0.5, X: [0, 0, 1] } })).toBe(true);
});
