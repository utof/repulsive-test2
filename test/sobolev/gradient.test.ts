import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { barycenterPhiAndC } from '../../src/core/sobolev/constraints';
import { solveConstrainedGradient } from '../../src/core/sobolev/gradient';
import { flatten } from '../../src/core/sobolev/layout';
import { calculateDisjointPairs } from '../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../src/core/testConfigs';

// All 5 oracle fixture/golden pairs (Stage-1 Sobolev oracle harness).
// @see oracle/README.md
const FIXTURE_NAMES = ['crossing', 'junction-y', 'helix', 'linked-rings', 'knot'] as const;

interface Fixture {
    name: string;
    vertices: Vec3[];
    edges: Edge[];
    alpha: number;
    beta: number;
    epsilon: number;
}

interface Golden {
    dE: Vec3[];
    x0_barycenter_target: Vec3;
    g_tilde_flat: number[];
    lambda: number[];
}

// Load at runtime (avoids needing resolveJsonModule in tsconfig; test/** is typechecked),
// mirroring test/sobolev/linsolve.test.ts.
function loadFixture(name: string): Fixture {
    return JSON.parse(
        readFileSync(new URL(`../../oracle/fixtures/${name}.json`, import.meta.url), 'utf8'),
    ) as Fixture;
}

function loadGolden(name: string): Golden {
    return JSON.parse(
        readFileSync(new URL(`../../oracle/golden/${name}.json`, import.meta.url), 'utf8'),
    ) as Golden;
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

for (const name of FIXTURE_NAMES) {
    const fixture = loadFixture(name);
    const golden = loadGolden(name);

    // DESIGN DECISION (preserve): the INPUTS here are the oracle's own outputs —
    // `golden.dE` as dE and `golden.x0_barycenter_target` as x0 — NOT a TS-side
    // finite-difference dE. This decouples the solve comparison from
    // cross-language finite-difference noise (energy-roundoff/h caps FD-vs-FD
    // agreement near ~1e-9 rel — see oracle/README.md "Known tolerance
    // caveats"), so the 1e-9 gates below measure ONLY assembly + solve.
    // dE-computation correctness is gated separately by oracle/compare_energy.ts.
    test(`solveConstrainedGradient: ${name} — matches oracle g̃/λ to 1e-9, residual ≤ 1e-10, descent, C·g̃ ≈ 0`, () => {
        const { vertices, edges, alpha, beta, epsilon } = fixture;
        const disjointPairs = calculateDisjointPairs(edges);
        const x0 = golden.x0_barycenter_target;

        const { gTilde, lambda, residual } = solveConstrainedGradient(
            vertices,
            edges,
            disjointPairs,
            alpha,
            beta,
            epsilon,
            golden.dE,
            x0,
        );
        const gFlat = flatten(gTilde);

        const gRelDiff =
            euclideanDiff(gFlat, golden.g_tilde_flat) / euclideanNorm(golden.g_tilde_flat);
        const lambdaRelDiff =
            euclideanDiff(lambda, golden.lambda) / Math.max(1, euclideanNorm(golden.lambda));
        const descentDot = dot(flatten(golden.dE), gFlat);
        // Constraint check recomputes C independently of the solve's internals.
        const { C } = barycenterPhiAndC(vertices, edges, x0);
        const constraintRel = euclideanNorm(matVec(C, gFlat)) / Math.max(1, euclideanNorm(gFlat));

        console.log(
            `[gradient] ${name} (|V| = ${vertices.length}): g̃ rel diff = ${gRelDiff.toExponential(3)}, ` +
                `λ rel diff = ${lambdaRelDiff.toExponential(3)}, residual = ${residual.toExponential(3)}, ` +
                `dEᵀg̃ = ${descentDot.toExponential(3)}, ‖C·g̃‖/max(1,‖g̃‖) = ${constraintRel.toExponential(3)}`,
        );

        // Cross-implementation gates vs the Python oracle.
        expect(gRelDiff).toBeLessThanOrEqual(1e-9);
        expect(lambdaRelDiff).toBeLessThanOrEqual(1e-9);
        // Self-certifying saddle residual — spec §E prop 8.
        // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §E (prop 8)
        expect(residual).toBeLessThanOrEqual(1e-10);
        // Descent positivity — spec §E prop 9. Guaranteed structurally: with
        // Cg̃ = 0, dEᵀg̃ = g̃ᵀĀg̃ ≥ 0 by PSD Ā, so a failure here means the
        // assembly or the solve is broken, not that the fixture is unlucky.
        // @see local_files/2026-07-02-sobolev-gradient-rsrch-results.md §E (prop 9)
        expect(descentDot).toBeGreaterThan(0);
        // g̃ lies in the null space of C — spec §E prop 8.
        expect(constraintRel).toBeLessThanOrEqual(1e-10);
    });
}
