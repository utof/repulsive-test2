import { expect, test } from 'bun:test';
import { existsSync, readFileSync } from 'node:fs';
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

test('makePreparedSaddle reproduces solveSaddleFromA on the PROJECTION rhs — the Woodbury/M path', () => {
    // THE GRADIENT RHS CANNOT TEST THIS, which is why this case is separate rather than
    // one more assertion above. The Woodbury correction of [DESIGN §4.1] step 3 is
    // `W·mu` with `mu = M⁻¹(Pᵀx̂)`, so it does nothing whenever `Pᵀx̂ = 0`. Measured at
    // n=60 (`scratch/t2-mechanism.ts`): `|Pᵀx̂|` is 3.7e-17 on the gradient rhs against
    // 8.6e-3 here — fourteen orders apart, so on a gradient rhs the ENTIRE Woodbury/M
    // path is multiplied by zero. Cause: `rhsBottom = 0` makes the Schur solution satisfy
    // `C x̂ = 0`, pinning the barycenter, and the tangent-point gradient is
    // translation-invariant (measured `|Pᵀb|/|b|` = 2.0e-16), so no null-space component
    // is left for Woodbury to restore. A projection rhs with nonzero barycenter rows
    // requires the barycenter to MOVE — exactly the null direction — so the term is live.
    // (Do NOT re-state this as "C's barycenter rows are Pᵀ up to scale": they are
    // length-weighted, with a measured relative weight spread of 1.01. It is `C·P` that
    // is a scaled `[I₃; 0]`, not the rows themselves.)
    // Three real defects pass every gradient-rhs test in this file and die on this one:
    // M formed in f32, the Woodbury sign flipped, and `residualOf` measuring against A_σ
    // instead of the unshifted A. Do not fold this into the test above and do not
    // "simplify" its rhs to the gradient shape.
    // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1 step 3
    const { A, C, n } = system(60);
    const { Asigma, sigma } = shiftSobolev(A, n);
    const L = choleskyF64(Asigma, n);
    const traces: IrTrace[] = [];
    const prepared = makePreparedSaddle(A, L, sigma, n, (t) => {
        traces.push(t);
    });
    const zeros = new Array<number>(3 * n).fill(0);
    const d = Array.from({ length: C.length }, (_, i) => -0.1 - 0.01 * i);
    const z = asExternal(prepared.factor(C)).solve([...zeros, ...d]);
    const ref = solveSaddleFromA(A, n, C, zeros, d);
    const zref = [...ref.x, ...ref.lambda];
    let num = 0;
    let den = 0;
    for (let i = 0; i < z.length; i++) {
        num += (z[i] - zref[i]) ** 2;
        den += zref[i] ** 2;
    }
    expect(Math.sqrt(num / den)).toBeLessThan(1e-10); // measured 8.9230e-15 at n=60

    // z₀ ALONE must clear the tolerance — assert steps 1–3, not steps 1–4. The comparison
    // above cannot see an M defect on its own: IR minimises the RESIDUAL, so it repairs a
    // degraded z₀ and hands back a converged z either way. Measured with M formed in f32:
    // the curve becomes [1.352e-8, 2.138e-15] — z₀ is six orders worse — yet the final
    // relative error is 8.1968e-15 against the correct 8.9230e-15, i.e. INDISTINGUISHABLE.
    // Only the refinement count separates them. This is the same steps-1–3-not-1–4
    // distinction gate CR is pinned to, and the reason it is pinned there.
    // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §5 (CR)
    expect(traces).toHaveLength(1);
    expect(traces[0].refinements).toBe(0);
    expect(traces[0].relResiduals).toHaveLength(1); // measured [1.124e-14] at n=60

    // Non-vacuity, in the shape D6 requires of CR's own projection row: assert the
    // Woodbury term is actually EXERCISED. Without it, a later change that quietly drove
    // `d` to zero would leave this test green while restoring the exact blindness it was
    // added to remove.
    // @see docs/superpowers/plans/2026-09-04-webgpu-solver-phase2a-1.md (## Decisions, D6)
    let ptx = 0;
    for (let a = 0; a < 3; a++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += z[a * n + i] / Math.sqrt(n);
        ptx += s * s;
    }
    expect(Math.sqrt(ptx)).toBeGreaterThan(1e-3); // measured 5.145e-2 at n=60
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
    // Pin the number of refinements the throw REPORTS HAVING APPLIED. This discriminates
    // an off-by-one only because `saddle.ts` interpolates the loop counter `it` into the
    // message; while it interpolated SADDLE_IR_MAX_REFINEMENTS the message read
    // "4 refinements" for every loop bound and this assertion was vacuous — measured, the
    // `it >= MAX - 1` mutant (budget z₀+3) left all nine tests in this file green. If a
    // future change moves the message back to the constant, delete this assertion rather
    // than leaving it to imply a guarantee it no longer carries.
    // Read from the constant, not a hard-coded `4`, so the pair stays in step if the
    // budget is ever re-calibrated ([CAL §E.5]/[CAL §E.6]).
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

/** Repo-relative, for the failure messages below. */
const K1CAL = 'bench/results/2026-09-04-gpu-phase2a-k1-calibration.json';
/**
 * The same artifact resolved relative to THIS FILE, for reading. Not the bare
 * repo-relative string: `readFileSync` resolves that against the CWD, so it would only
 * work when `bun test` is invoked from the repo root. Same idiom `test/gpu/cr.test.ts:58`
 * uses for the same artifact.
 */
const K1CAL_URL = new URL(`../../${K1CAL}`, import.meta.url);

/**
 * `t4` — the pre-registered IR tolerance and budget — READ from the committed [K1CAL]
 * artifact, never transcribed. [CAL §A]'s binding rule is that no gate number lives in
 * prose unless a committed script emitted it into a committed JSON; a literal table here
 * would be that same failure one level down.
 *
 * LAZY, and that is load-bearing, for the reason `test/gpu/cr.test.ts:71` gives: read at
 * module scope, a missing or moved artifact throws during IMPORT and aborts the whole file
 * before any `test()` registers, so the failure would surface as an unrelated import error
 * rather than as the named test below.
 * @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-gate-calibration.md §A, §E.5
 */
function calibrationT4(): { tol: number; maxIters: number } {
    if (!existsSync(K1CAL_URL))
        throw new Error(
            `the [K1CAL] artifact is missing at ${K1CAL_URL.pathname} — the IR budget and ` +
                'tolerance cannot be checked against the calibration that pre-registered them',
        );
    const parsed = JSON.parse(readFileSync(K1CAL_URL, 'utf8')) as {
        t4?: { tol: number; maxIters: number };
    };
    if (!parsed.t4)
        throw new Error(
            `the [K1CAL] artifact at ${K1CAL} has no t4 block — [CAL §E.5]'s pre-registered ` +
                'IR budget has no committed source and this pin would be vacuous',
        );
    return parsed.t4;
}

test("SADDLE_IR_MAX_REFINEMENTS and SADDLE_IR_TOL ARE [CAL]'s committed t4 block", () => {
    // WHAT BREAKS IF EITHER NUMBER MOVES, and why nothing else in the suite notices.
    // [CAL §E.5] measures that an f32 factor sitting exactly AT gate K1's green bar
    // (f = k1Bar/alpha = 3.334) needs all four refinements with ZERO spare at n=960. So a
    // silent 4 -> 3 makes every legitimate GPU factor throw ExternalSolveError at n=960,
    // sending every step down the CPU fallback with `usedGpuSolve: false`: the P2a speedup
    // evaporates and NOTHING goes red. Measured on this branch: the 4 -> 3 mutant left all
    // 325 tests passing. `makePreparedSaddle throws ExternalSolveError when IR cannot
    // converge` cannot pin it either — that assertion READS the constant (deliberately;
    // that is what makes it discriminate the loop-condition off-by-one), so it moves with
    // the value. Changing either number is a spec change, not a tuning knob.
    //
    // BOTH halves are load-bearing. The literals catch a silent edit to `src/gpu/saddle.ts`.
    // The equality against the artifact catches the other direction — [CAL] re-run with a
    // different `T4_MAX_ITERS` (`bench/gpu/k1-calibration.py:104`, emitted into the JSON at
    // `bench/gpu/k1-calibration.py:471`) while the TypeScript keeps the old number, a
    // divergence a transcribed literal cannot see.
    // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-gate-calibration.md §E.5, §E.6, §A
    // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1 step 4, §4.2
    expect(SADDLE_IR_MAX_REFINEMENTS).toBe(4);
    expect(SADDLE_IR_TOL).toBe(1e-10);
    const t4 = calibrationT4();
    expect(SADDLE_IR_MAX_REFINEMENTS).toBe(t4.maxIters);
    expect(SADDLE_IR_TOL).toBe(t4.tol);
});

/**
 * An EXACT rescale: a power of two, so `v * IR_SCALE` is exact in binary floating point at
 * every magnitude here and the whole solve — which is linear in the rhs with rhs-independent
 * coefficients — scales bit-for-bit. Do NOT "simplify" this to 1e-6: an inexact factor
 * perturbs the last bits of a residual already sitting at its own cancellation floor, and
 * the two curves then differ in the 4th significant digit on CORRECT code. Measured: 1.0008
 * against an expected 1.
 */
const IR_SCALE = 2 ** -20;
/** As above, driven far enough down that the ABSOLUTE residual falls under the tolerance. */
const IR_DEEP_SCALE = 2 ** -60;

test('the IR criterion is scale-free: the residual curve is BIT-IDENTICAL under an exact rhs rescale', () => {
    // The criterion at `src/gpu/saddle.ts:412` is the PURE relative residual
    // `||r_i||/||r||`, and `||r_i||/max(1, ||r||)` — the spelling [DESIGN §4.1] step 4
    // forbids in bold and §4.4 forbids again — is scale-DEPENDENT below ||r|| = 1, which
    // the projection rhs `[0; -Phi]` always is. Nothing in the suite bound that before this
    // test: measured, the `max(1, ||r||)` mutant left all 325 tests passing, because every
    // GATED CR row that exercises IR is gradient-shaped (||r|| >> 1, where the two
    // spellings agree) and the projection-shaped rows that would move are ungated.
    // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1 step 4, §4.4
    const { A, C, n } = system(60);
    const { Asigma, sigma } = shiftSobolev(A, n);
    const L = choleskyF64(Asigma, n);
    // Degrade the factor so z0 does NOT already clear the tolerance and IR walks a real
    // curve. With an exact f64 factor the curve is ONE entry long at every scale and this
    // test would exercise a single evaluation of the criterion. 1e-5 is measured to give 2
    // refinements at n=60 — a three-entry curve, two spare against the budget of 4.
    for (let i = 0; i < n; i++) L[i * n + i] *= 1 + 1e-5 * ((i % 5) - 2);
    const traces: IrTrace[] = [];
    const prepared = makePreparedSaddle(A, L, sigma, n, (t) => {
        traces.push(t);
    });
    const fac = asExternal(prepared.factor(C));
    const zeros = new Array<number>(3 * n).fill(0);
    // The PROJECTION rhs shape, not the gradient one: `||rhs||` must be < 1 for
    // `max(1, ||rhs||)` to differ from `||rhs||` at all. Measured 2.3108e-1 at n=60.
    const d = Array.from({ length: C.length }, (_, i) => -0.1 - 0.01 * i);
    let rhsNorm = 0;
    for (const v of d) rhsNorm += v * v;
    expect(Math.sqrt(rhsNorm)).toBeLessThan(1);

    fac.solve([...zeros, ...d]);
    fac.solve([...zeros, ...d.map((v) => v * IR_SCALE)]);

    expect(traces).toHaveLength(2);
    // Non-vacuity: IR must actually fire, or the "curve" is one entry and the comparison
    // below says nothing about the criterion's later evaluations. Measured 2 at n=60.
    expect(traces[0].refinements).toBeGreaterThan(0);
    expect(traces[1].refinements).toBe(traces[0].refinements);
    // BIT-identical, not a tolerance band: the rescale is exact (see IR_SCALE), so under
    // the pure relative criterion every entry is the SAME f64. Measured green at HEAD. On
    // the `max(1, ||rhs||)` mutant the run goes red one line earlier, at the refinement
    // count: the rescaled solve's ABSOLUTE residual (measured 6.2144e-11) is already under
    // the 1e-10 tolerance, so IR converges at iteration 0 and returns a z that has had NO
    // refinement applied — 0 against the unscaled run's 2.
    expect(traces[1].relResiduals).toEqual(traces[0].relResiduals);
});

test('the IR criterion rejects an unconvergeable factor at EVERY rhs scale', () => {
    // The consequence [DESIGN §4.1] step 4 actually needs: "an unconverged z must never
    // reach the line search", at any rhs magnitude. Under `max(1, ||rhs||)` the criterion
    // is ABSOLUTE for `||rhs|| <= 1`, so shrinking the projection rhs far enough makes a
    // factor with a relative residual of 3.06e+16 look converged at iteration 0 and
    // `solve()` RETURNS that garbage instead of throwing. That is the permissive direction
    // the TSDoc at `src/gpu/saddle.ts:35-40` calls "~1e4x worse than this criterion claims",
    // exhibited as a contract breach rather than as a number.
    // @see docs/superpowers/specs/2026-09-03-webgpu-solver-phase2-design.md §4.1 step 4, §4.4
    const { A, C, n } = system(60);
    const { Asigma, sigma } = shiftSobolev(A, n);
    const L = choleskyF64(Asigma, n);
    // The same corruption `makePreparedSaddle throws ExternalSolveError when IR cannot
    // converge` uses: IR's contraction exceeds 1, so no budget can rescue it.
    for (let i = 0; i < n; i++) L[i * n + i] *= 1 + 0.3 * ((i % 7) - 3);
    const traces: IrTrace[] = [];
    const prepared = makePreparedSaddle(A, L, sigma, n, (t) => {
        traces.push(t);
    });
    const fac = asExternal(prepared.factor(C));
    const zeros = new Array<number>(3 * n).fill(0);
    const d = Array.from({ length: C.length }, (_, i) => -0.1 - 0.01 * i);

    const messages = [1, IR_DEEP_SCALE].map((s) => {
        try {
            fac.solve([...zeros, ...d.map((v) => v * s)]);
            return '(returned — the solve was ACCEPTED)';
        } catch (e) {
            expect(e).toBeInstanceOf(ExternalSolveError);
            return (e as Error).message;
        }
    });
    // Non-vacuity in BOTH directions. First: the unscaled solve must already throw, or the
    // corruption has stopped being hopeless and the scaled case proves nothing.
    expect(messages[0]).toContain('saddle IR: relative residual');
    // Second: at IR_DEEP_SCALE the ABSOLUTE residual — `rel * ||rhs||`, which is exactly
    // what `max(1, ||rhs||)` would compare against the tolerance at iteration 0 — must sit
    // BELOW the tolerance, or the mutant would throw here too and the test would be green
    // for the wrong reason. Measured 8.3e-18 against a bar of 1e-10.
    let deepNorm = 0;
    for (const v of d) deepNorm += (v * IR_DEEP_SCALE) ** 2;
    expect(traces[1].relResiduals[0] * Math.sqrt(deepNorm)).toBeLessThan(SADDLE_IR_TOL);
    // The criterion is scale-free, so the rejection is bit-for-bit the same event: same
    // relative residual to every printed digit, same refinement count.
    expect(messages[1]).toBe(messages[0]);
});
