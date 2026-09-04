import { expect, test } from 'bun:test';
import { trefoil } from '../../src/core/fixtures';
import { DEFAULTS } from '../../src/core/optimizer';
import {
    barycenterBlock,
    evaluateConstraintSet,
    totalLengthBlock,
} from '../../src/core/sobolev/constraintSet';
import { assembleAFlat } from '../../src/core/sobolev/innerProduct';
import { flatten } from '../../src/core/sobolev/layout';
import { ExternalSolveError, solveSaddleFromA } from '../../src/core/sobolev/linsolve';
import { calculateDisjointPairs, gradientAnalytical } from '../../src/core/tangentPointEnergy';
import type { Vec3 } from '../../src/core/testConfigs';
import {
    asExternal,
    choleskyF64,
    type IrTrace,
    makePreparedSaddle,
    SADDLE_IR_MAX_REFINEMENTS,
    SADDLE_IR_TOL,
    shiftSobolev,
} from '../../src/gpu/saddle';

function sobolevA(n: number): { A: Float64Array; n: number } {
    const { vertices, edges } = trefoil(n);
    const dp = calculateDisjointPairs(edges);
    return {
        A: assembleAFlat(vertices, edges, dp, DEFAULTS.alpha, DEFAULTS.beta, DEFAULTS.epsilon),
        n,
    };
}

test('choleskyF64 reconstructs A_σ to f64 round-off', () => {
    const { A, n } = sobolevA(60);
    const { Asigma } = shiftSobolev(A, n);
    const L = choleskyF64(Asigma, n);
    let maxAbs = 0;
    let maxErr = 0;
    for (let i = 0; i < n; i++)
        for (let j = 0; j <= i; j++) {
            let s = 0;
            for (let k = 0; k <= j; k++) s += L[i * n + k] * L[j * n + k];
            maxErr = Math.max(maxErr, Math.abs(s - Asigma[i * n + j]));
            maxAbs = Math.max(maxAbs, Math.abs(Asigma[i * n + j]));
        }
    expect(maxErr / maxAbs).toBeLessThan(1e-12);
});

test('the shift is positive exactly on null(A): A_σ(𝟙/√n) = σ(𝟙/√n)', () => {
    // [DESIGN §4.1] step 1 rests on this. A 𝟙 ≈ 0 (null(A) = span(𝟙), one-dimensional
    // even for the disconnected presets, [DESIGN §2]), so A_σ 𝟙 = (σ/n)·𝟙𝟙ᵀ𝟙 = σ𝟙.
    const { A, n } = sobolevA(60);
    const { Asigma, sigma } = shiftSobolev(A, n);
    const one = new Float64Array(n).fill(1 / Math.sqrt(n));
    let maxDev = 0;
    for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += Asigma[i * n + j] * one[j];
        maxDev = Math.max(maxDev, Math.abs(s - sigma * one[i]));
    }
    expect(maxDev / sigma).toBeLessThan(1e-10);
});

// The pivot guard is tested on SYNTHETIC 2×2 matrices with exactly-representable
// entries — NEVER on the real unshifted A.
//
// Why the obvious test is impossible: A is PSD with a ONE-dimensional null space
// ([DESIGN §2]), so its terminal Schur complement is 0 ± roundoff and the SIGN of that
// roundoff is an accident, not a property. Measured on this branch: n = 8 / 24 / 60 / 61
// / 62 complete with NO throw — at n=60 choleskyF64 returns having written
// L[59·60+59] = sqrt(2.49e-14) = 1.58e-7, i.e. it silently produces exactly the garbage
// factor the guard exists to reject; n=120 throws d = −3.062e-12 at column 119 and n=240
// throws d = −6.537e-12 at column 239. An assertion pinned to any of those is pinned to a
// rounding accident: a change to assembleAFlat, to trefoil, or to the toolchain flips it,
// and the test then either goes red against a CORRECT guard (whose two natural "repairs"
// both corrupt production code — widening `!(d > 0)` to a tolerance rejects legitimately
// tiny-but-positive pivots on A_σ, and changing the shift breaks [DESIGN §4.1] step 1) or
// silently stops testing anything. The four cases below are exact in f64 and depend on
// nothing.
// @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3 (the GPU
//      `finalize` kernel applies the same `!(d > 0)` spelling)
test('choleskyF64 throws on a negative pivot', () => {
    // [[1,2],[2,1]] — eigenvalues 3 and −1. Column 0 passes (d = 1); column 1 has
    // d = 1 − 2² = −3 exactly, no rounding involved. Pinning the value and the column
    // means a guard that fired for the wrong reason would not pass this.
    expect(() => choleskyF64(new Float64Array([1, 2, 2, 1]), 2)).toThrow(
        /non-positive pivot -3 at column 1/,
    );
});

test('choleskyF64 throws on an exactly-zero pivot', () => {
    // [[0,0],[0,1]] — the PSD boundary, reached exactly rather than by roundoff. This is
    // the case the real A only ever approaches, which is why it is synthesised here.
    // Without the guard, d = 0 divides and writes Infinity/NaN into L.
    expect(() => choleskyF64(new Float64Array([0, 0, 0, 1]), 2)).toThrow(
        /non-positive pivot 0 at column 0/,
    );
});

test('choleskyF64 throws on a NaN pivot — the case `d <= 0` would let through', () => {
    // THE reason the guard is spelt `!(d > 0)` and not `d <= 0`: `NaN <= 0` is FALSE, so
    // the `d <= 0` spelling admits NaN, takes sqrt(NaN) = NaN, and silently poisons every
    // later backsolve; `!(NaN > 0)` is TRUE. This test is the only thing in the suite that
    // can catch that "simplification" — no fixture produces a NaN pivot by accident, and
    // the n=120 fixture test an earlier draft used could not have detected it either.
    // Do NOT merge it into the negative-pivot case and do NOT delete it.
    // (The guard's second half, `!Number.isFinite(d)`, covers +Infinity, which `!(d > 0)`
    // alone would pass.)
    // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.3
    expect(() => choleskyF64(new Float64Array([Number.NaN, 0, 0, 1]), 2)).toThrow(
        /non-positive pivot NaN at column 0/,
    );
});

test('choleskyF64 does NOT throw on an SPD matrix (the positive control)', () => {
    // Without this, the three tests above all pass against a guard that throws
    // unconditionally — i.e. against a completely broken factorization.
    // [[4,1],[1,3]]: L = [[2,0],[0.5,sqrt(2.75)]].
    const L = choleskyF64(new Float64Array([4, 1, 1, 3]), 2);
    expect(L[0]).toBeCloseTo(2, 12);
    expect(L[2]).toBeCloseTo(0.5, 12);
    expect(L[3]).toBeCloseTo(Math.sqrt(2.75), 12);
});

function system(n: number) {
    const { vertices, edges } = trefoil(n);
    const dp = calculateDisjointPairs(edges);
    const { alpha, beta, epsilon } = DEFAULTS;
    const A = assembleAFlat(vertices, edges, dp, alpha, beta, epsilon);
    const x0: Vec3 = [0, 0, 0];
    for (const v of vertices) {
        x0[0] += v[0] / n;
        x0[1] += v[1] / n;
        x0[2] += v[2] / n;
    }
    const L0 = edges.reduce(
        (s, [a, b]) =>
            s +
            Math.hypot(
                vertices[a][0] - vertices[b][0],
                vertices[a][1] - vertices[b][1],
                vertices[a][2] - vertices[b][2],
            ),
        0,
    );
    const set = [barycenterBlock(x0), totalLengthBlock(L0)];
    const { C } = evaluateConstraintSet(set, vertices, edges);
    const dE = gradientAnalytical(vertices, edges, dp, alpha, beta, epsilon);
    return { vertices, edges, A, C, rhsTop: flatten(dE), n };
}

test('makePreparedSaddle with an EXACT f64 factor reproduces solveSaddleFromA', () => {
    const { A, C, rhsTop, n } = system(60);
    const { Asigma, sigma } = shiftSobolev(A, n);
    const L = choleskyF64(Asigma, n);
    const prepared = makePreparedSaddle(A, L, sigma, n);
    // asExternal, not `.factor(C).solve(...)`: factor() returns the WIDENED
    // SaddleFactorization union ([DESIGN §4.4] pins that signature) and `solve` exists
    // on only one of its three members, so the direct spelling is a TS2339 that stops
    // the plan's `bunx tsc --noEmit &&` chain before `bun test`.
    const z = asExternal(prepared.factor(C)).solve([...rhsTop, ...new Array(C.length).fill(0)]);
    const ref = solveSaddleFromA(A, n, C, rhsTop);
    const zref = [...ref.x, ...ref.lambda];
    let num = 0;
    let den = 0;
    for (let i = 0; i < z.length; i++) {
        num += (z[i] - zref[i]) ** 2;
        den += zref[i] ** 2;
    }
    // A smoke bar, not the gate: CR is the κ-relative pre-registered one.
    expect(Math.sqrt(num / den)).toBeLessThan(1e-10);
});

test('makePreparedSaddle throws ExternalSolveError when IR cannot converge', () => {
    const { A, C, rhsTop, n } = system(60);
    const { Asigma, sigma } = shiftSobolev(A, n);
    const L = choleskyF64(Asigma, n);
    // A factor corrupted far past any backward error a real kernel produces: IR's
    // contraction ‖I − M⁻¹K‖ exceeds 1, so it cannot converge at any budget.
    for (let i = 0; i < n; i++) L[i * n + i] *= 1 + 0.3 * ((i % 7) - 3);
    const prepared = makePreparedSaddle(A, L, sigma, n);
    const fac = asExternal(prepared.factor(C));
    let err: unknown;
    try {
        fac.solve([...rhsTop, ...new Array(C.length).fill(0)]);
    } catch (e) {
        err = e;
    }
    expect(err).toBeInstanceOf(ExternalSolveError);
    // Pin the budget the throw reports, from the constant rather than a literal: a
    // silent off-by-one in the loop's counting understates every bar derived from it
    // ([CAL §E.5]/[CAL §E.6]), and a hard-coded `4` here would not catch that.
    expect((err as Error).message).toContain(`${SADDLE_IR_MAX_REFINEMENTS} refinements`);
});

test('the IR trace records the convergence curve', () => {
    const { A, C, rhsTop, n } = system(60);
    const { Asigma, sigma } = shiftSobolev(A, n);
    const L = choleskyF64(Asigma, n);
    // An ARRAY, NOT `let trace: IrTrace | null = null`. TypeScript does not account for
    // assignments made inside a callback when narrowing a `let` in the enclosing scope
    // (microsoft/TypeScript#9998), so `trace` stays narrowed to `null` and every
    // `trace?.x` becomes a property access on `never` — two TS2339s under this repo's
    // tsconfig, which stops the plan's `bunx tsc --noEmit &&` chain before `bun test`.
    // `?.` does not rescue it; `?.` is what produces the `never`. Array mutation is not
    // narrowed away, and `traces` being `const` means no CFA reset applies.
    const traces: IrTrace[] = [];
    const prepared = makePreparedSaddle(A, L, sigma, n, (t) => {
        traces.push(t);
    });
    asExternal(prepared.factor(C)).solve([...rhsTop, ...new Array(C.length).fill(0)]);
    expect(traces).toHaveLength(1);
    // The EXACT count, not `<= SADDLE_IR_MAX_REFINEMENTS`. `solve()` returns only when
    // `rel <= SADDLE_IR_TOL` and otherwise throws, so `refinements <= 4` and
    // `relResiduals.at(-1) <= 1e-10` are both true BY CONSTRUCTION whenever the call
    // returned — they assert nothing. With an exact f64 factor at n=60, z₀ already
    // clears the tolerance, so IR applies ZERO corrections and the curve has one entry;
    // that is the falsifiable claim. (At n=960 the same measurement is 1 — the number
    // this test would notice changing.)
    expect(traces[0].refinements).toBe(0);
    expect(traces[0].relResiduals).toHaveLength(1);
    expect(traces[0].relResiduals[0]).toBeLessThanOrEqual(SADDLE_IR_TOL);
});
