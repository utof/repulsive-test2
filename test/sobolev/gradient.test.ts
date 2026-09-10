import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { barycenterBlock } from '../../src/core/sobolev/constraintSet';
import { barycenterPhiAndC } from '../../src/core/sobolev/constraints';
import {
    solveConstrainedGradient,
    solveConstrainedGradientSetFrozen,
} from '../../src/core/sobolev/gradient';
import { assembleAFlat } from '../../src/core/sobolev/innerProduct';
import { flatten } from '../../src/core/sobolev/layout';
import {
    luSolve,
    type PreparedSaddle,
    type SaddleFactorization,
} from '../../src/core/sobolev/linsolve';
import {
    type SobolevStepTimings,
    timingsBegin,
    timingsEnd,
} from '../../src/core/sobolev/phaseTimings';
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

// ─── Phase-ledger contract of the `saddle?` (WebGPU Phase 2a) seam path ───────────
//
// `solveConstrainedGradientSetFrozen`'s two arms both wrap their solve in
// `timed('saddle', …)` — gradient.ts:143 for the default CPU arm, gradient.ts:132 for
// the PreparedSaddle arm. Only the CPU one was pinned by a test (test/gpu/
// saddleSeam.test.ts:87 asserts `saddle.calls >= 1` on a seam-ABSENT step), so the
// seam arm's key was free to drift to any other string without a red test. The key IS
// the bench ledger's schema (see the "Add keys; never rename" note in
// src/core/sobolev/phaseTimings.ts) — a drifted key silently drops the GPU path's
// solve out of every bench comparison against the CPU baseline instead of erroring.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.4
// @see docs/superpowers/plans/2026-07-03-sobolev-solver-perf.md (Task 1)

/**
 * A {@link PreparedSaddle} that hands back the SAME K the core would have built, LU'd
 * on the spot — the routing probe, not a numerics probe (the real factor is the GPU
 * Cholesky). `factorCalls` is exposed so a test can prove the seam arm was actually
 * taken rather than silently falling through to the CPU arm. Deliberately calls no
 * `timed()` of its own: the ledger assertion below is exact, so a timing wrap here
 * would be indistinguishable from the one under test.
 * Note the coordinate-major block write `K[b*n+i][b*n+j]` — Ā = I₃ ⊗ A.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1, §4.4
 */
function preparedFromCore(
    f: Fixture,
    disjointPairs: number[][],
): PreparedSaddle & {
    factorCalls: () => number;
} {
    const A = assembleAFlat(f.vertices, f.edges, disjointPairs, f.alpha, f.beta, f.epsilon);
    let calls = 0;
    return {
        A,
        factorCalls: () => calls,
        factor(C: number[][]): SaddleFactorization {
            calls++;
            const n = f.vertices.length;
            const m = 3 * n;
            const k = C.length;
            const size = m + k;
            const K: number[][] = Array.from({ length: size }, () => new Array(size).fill(0));
            for (let b = 0; b < 3; b++)
                for (let i = 0; i < n; i++)
                    for (let j = 0; j < n; j++) K[b * n + i][b * n + j] = A[i * n + j];
            for (let r = 0; r < k; r++)
                for (let c = 0; c < m; c++) {
                    K[m + r][c] = C[r][c];
                    K[c][m + r] = C[r][c];
                }
            return { kind: 'external', solve: (rhs: number[]) => luSolve(K, rhs) };
        },
    };
}

test('solveConstrainedGradientSetFrozen: the PreparedSaddle arm records its solve under the `saddle` phase key', () => {
    const fixture = loadFixture('crossing');
    const golden = loadGolden('crossing');
    const disjointPairs = calculateDisjointPairs(fixture.edges);
    // Built BEFORE arming: assembleAFlat is itself wrapped in timed('assembleA') with
    // 'bHigh'/'bLow' sub-phases (innerProduct.ts:336-341), and this test's ledger
    // assertion is an exact key set. On the real seam path A likewise arrives
    // pre-assembled from the GPU, so this ordering is the honest one.
    const prepared = preparedFromCore(fixture, disjointPairs);

    let ledger: SobolevStepTimings | null = null;
    let out: ReturnType<typeof solveConstrainedGradientSetFrozen>;
    timingsBegin();
    try {
        out = solveConstrainedGradientSetFrozen(
            fixture.vertices,
            fixture.edges,
            disjointPairs,
            fixture.alpha,
            fixture.beta,
            fixture.epsilon,
            golden.dE,
            [barycenterBlock(golden.x0_barycenter_target)],
            undefined,
            prepared,
        );
    } finally {
        // Always disarm: a leaked `acc` would silently pollute every later test in the
        // process (the collector is module-scoped, phaseTimings.ts:40).
        ledger = timingsEnd();
    }

    // Non-vacuity, in three independent ways — without these the ledger assertions
    // below would still pass over a solve that never happened or never took the seam.
    expect(prepared.factorCalls()).toBe(1);
    expect(out.residual).toBeLessThanOrEqual(1e-10);
    expect(
        euclideanDiff(flatten(out.gTilde), golden.g_tilde_flat) /
            euclideanNorm(golden.g_tilde_flat),
    ).toBeLessThanOrEqual(1e-9);

    // The contract. solveSaddleFrozen deliberately fires no 'factor' sub-phase
    // (linsolve.ts:924-925 — "no factorization; that is the point of the reuse") and
    // evaluateConstraintSet is untimed, so gradient.ts:132 is the ONE timed() call on
    // this arm and the ledger must be exactly {saddle: 1 call}.
    expect(Object.keys(ledger ?? {})).toEqual(['saddle']);
    expect(ledger?.saddle?.calls).toBe(1);
});
