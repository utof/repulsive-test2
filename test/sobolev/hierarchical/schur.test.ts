/**
 * Spec §8.2 / §11.2 row 9b of Phase 2H: `schur.ts`, the `k = O(1)` constraint route -- the FULL
 * saddle `[[Ā, Cᵀ], [C, 0]]` solved by matrix-free GMRES on `A_σ` plus a small dense Schur
 * complement. The oracle is the dense path (`assembleAFlat` / `solveSaddleFromA`, LDLᵀ), which
 * shares NO code with the hierarchical stack (different algorithm, different operator path).
 *
 * WHAT THE ORACLE CAN AND CANNOT CERTIFY, MEASURED (this is why the cells below carry TWO columns):
 *  1. The dense LDLᵀ oracle is itself 1.4e-10..6.4e-10 from the TRUE solution at n = 960 (truth by
 *     double-double iterative refinement, which converges in one step to |dz|/|z| ~ 5e-17). So
 *     "agrees with the dense path to 1e-10 at n = 960" is BELOW the oracle's own error: Schur's
 *     agreement with it plateaus at that error whatever `tol` is. Every n = 960 cell therefore
 *     asserts the 1e-10 bar against the REFINED TRUTH of the dense matrix and merely LOGS the literal
 *     agreement with the LDLᵀ solution. At n = 240 the dense error is ~1e-12, so the literal bar is
 *     asserted there as well.
 *  2. The BCT operator at `DEFAULT_THETA` is not Ā: its solve differs from the dense Ā-solve by
 *     2e-3..6e-3 (the θ compression, spec H1) whatever the solver does. So the oracle for the
 *     SOLVER is the dense matrix of the SAME operator (`densify(applyAScalar)`); the independent
 *     operator path (θ = 0 streaming vs `assembleAFlat`) is exercised at n = 240 only, because a
 *     streaming apply is O(N²) (n = 960: 88 s to prepare a k = 3 solver, measured).
 *
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.2, §8.3, §11.2 row 9b, A33
 * @issue utof/repulsive-test2#62
 */
import { describe, expect, setDefaultTimeout, test } from 'bun:test';
import {
    barycenterBlock,
    edgeLengths,
    edgeLengthsBlock,
    evaluateConstraintSet,
    pointBlock,
    totalLengthBlock,
} from '../../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../../src/core/sobolev/constraints';
import { DEFAULT_THETA } from '../../../src/core/sobolev/hierarchical/bct';
import type { Preconditioner } from '../../../src/core/sobolev/hierarchical/precond';
import {
    DEFAULT_ETA_ACCEPT,
    DEFAULT_SCHUR_TOL,
    prepareSchur,
    RANK_PIVOT_REL,
    SCHUR_MAX_K,
    SchurConvergenceError,
    type SchurDefect,
    SchurRankError,
    SchurRouteError,
    TRANSLATION_INVARIANT_REL,
} from '../../../src/core/sobolev/hierarchical/schur';
import {
    ExternalSolveError,
    SolveNotCertifiedError,
    solveSaddleFromA,
    solveSaddleFrozen,
} from '../../../src/core/sobolev/linsolve';
import { fullTest } from '../../tier';
import { cholesky, cholSolve, spectrum } from './denseSpectrum';
import {
    BAR,
    checkCells,
    demean,
    denseRefs,
    dist,
    fixture,
    lcgVec,
    log,
    relErr,
    SET_NAMES,
    type SetName,
    setup,
    totalLen,
    UNPINNED,
    zeros,
} from './schurHarness';

setDefaultTimeout(900_000);

// ===========================================================================
// 1. The oracle cells: Schur vs the dense saddle.
// ===========================================================================

describe('Schur vs dense: same operator (lists at DEFAULT_THETA), spec §11.2 row 9b', () => {
    for (const name of SET_NAMES) {
        // n = 240: the dense error is ~1e-12, so the LITERAL 1e-10 bar is meaningful here.
        test(`n=240 ${name}: literal 1e-10 vs dense LDLᵀ, and vs refined truth`, () => {
            checkCells(240, DEFAULT_THETA, 'op', name, { literalBar: true, truthBar: true });
        });
    }
    // The n = 960 cells (full tier) are in schur.n960.test.ts, with the same describe name.
});

describe('Schur vs dense: independent operator path (θ = 0 streaming vs assembleAFlat), n = 240', () => {
    // Why n = 240 only: a streaming apply is O(N²); at n = 960 preparing a k = 3 solver took 88 s and
    // one solve 22-33 s (MEASURED), so the suite cannot afford it. The n = 960 θ = 0 numbers are in
    // the task report: Schur is 1.7e-10 / 2.1e-10 from the Ā-truth there (the stream apply and
    // `assembleAFlat` differ by relFro 1.4e-15 at n = 480, ~12 ulp, and x moves 6-9e-12 per ulp-sized
    // random perturbation at n = 960).
    // DEFAULT TIER runs `pin` (pinned) and `len` (unpinned); the other four sets are full-tier
    // (test/tier.ts), registered so they count as skipped. Why: an O(N²) streaming apply per matvec
    // (~29 s for the four). The lists operator checks every set at n = 240 in the default tier
    // anyway; what waits for a full run is only this cross-check against an operator path that
    // shares no code, on `k0`, `bary`, `barylen`, `barypin` (README.md "Tests").
    const STREAMING_DEFAULT: ReadonlySet<SetName> = new Set(['pin', 'len']);
    for (const name of SET_NAMES) {
        (STREAMING_DEFAULT.has(name) ? test : fullTest)(
            `n=240 ${name}: streaming operator vs assembleAFlat`,
            () => {
                checkCells(240, 0, 'A', name, { literalBar: true, truthBar: true });
            },
        );
    }
});

// ===========================================================================
// 2. tol: the measurement behind DEFAULT_SCHUR_TOL, and the gate's non-vacuity.
// ===========================================================================

// The constant the sweep below justifies stays pinned in the DEFAULT tier, so changing it cannot
// slip past a default run even though the sweep itself waits for the full tier.
test('DEFAULT_SCHUR_TOL is 1e-12, the knee the full-tier tol sweep measures', () => {
    expect(DEFAULT_SCHUR_TOL).toBe(1e-12);
});

// The tol sweep itself (full tier) is in schur.n960.test.ts.

// ===========================================================================
// 3. σ-independence: the A_σ shift must cancel (trap 1).
// ===========================================================================

describe('the shift cancels: x does not depend on σ (the Ā vs A_σ trap)', () => {
    for (const name of ['pin', 'barylen'] as const) {
        test(`n=240 ${name}: σ ×1/4/0.25 give the same x; the naive-shift knock-out does not`, () => {
            const { fx, C, cells } = setup(240, DEFAULT_THETA, name);
            const solveAt = (s: number, defect?: SchurDefect) => {
                const { op, P } = fx.withSigmaScale(s);
                const solver = prepareSchur(op, P, C, defect ? { knockOut: true, defect } : {});
                return cells.map((rhs) => solver.solve(rhs.top, rhs.bottom).x);
            };
            const base = solveAt(1);
            for (const s of [4, 0.25]) {
                const other = solveAt(s);
                for (let c = 0; c < cells.length; c++) {
                    const e = relErr(other[c], base[c]);
                    log(`σ×${s} ${name} ${cells[c].name}`, e.toExponential(2));
                    expect(e).toBeLessThan(BAR);
                }
            }
            // Non-vacuity: the same comparison on the naive Schur complement moves a lot.
            const naive1 = solveAt(1, 'naive-shift');
            const naive4 = solveAt(4, 'naive-shift');
            let worst = 0;
            for (let c = 0; c < cells.length; c++)
                worst = Math.max(worst, relErr(naive4[c], naive1[c]));
            log(`σ×4 ${name} naive-shift worst`, worst.toExponential(2));
            expect(worst).toBeGreaterThan(1e-5);
        });
    }
});

// ===========================================================================
// 4. The prepare/solve split, as structural integers.
// ===========================================================================

/** A preconditioner wrapper that COUNTS applies: the integer a fourth GMRES per solve would change. */
function counting(P: Preconditioner) {
    const c = { applies: 0 };
    return {
        c,
        P: {
            n: P.n,
            sigma: P.sigma,
            apply: (R: ArrayLike<number>, out: Float64Array) => {
                c.applies++;
                P.apply(R, out);
            },
        },
    };
}

describe('prepare pays k GMRES once; every solve is ONE (spec §8.2, the reason `prepare` exists)', () => {
    test('n=240 barypin (k=7): prepare = Σ(iters+1) applies, each solve = iters+1, zero rhs = 0', () => {
        const { fx, C, cells, k } = setup(240, DEFAULT_THETA, 'barypin');
        expect(k).toBe(7);
        const { c, P } = counting(fx.P);
        const solver = prepareSchur(fx.op, P, C);
        const prepareApplies = solver.report.columns.reduce((a, r) => a + r.precondApplies, 0);
        expect(c.applies).toBe(prepareApplies);
        expect(c.applies).toBe(solver.report.columns.reduce((a, r) => a + r.iters + 1, 0));
        const before = c.applies;
        const grad = cells[0];
        const r1 = solver.solve(grad.top, grad.bottom);
        expect(c.applies - before).toBe(r1.gmres.iters + 1);
        // A second, different rhs does not re-run prepare: the cost is again ONE GMRES, not k.
        const gen = cells[2];
        const mid = c.applies;
        const r2 = solver.solve(gen.top, gen.bottom);
        expect(c.applies - mid).toBe(r2.gmres.iters + 1);
        expect(c.applies - mid).toBeLessThan(prepareApplies / 3);
        // The projection solve (rhsTop = 0) is a back-substitution only: no GMRES at all.
        const zero = cells[1];
        const z0 = c.applies;
        const rz = solver.solve(zero.top, zero.bottom);
        expect(c.applies).toBe(z0);
        expect(rz.gmres.stop).toBe('zero-rhs');
        expect(rz.gmres.iters).toBe(0);
        log('barypin prepare applies / solve applies', `${prepareApplies} / ${r1.gmres.iters + 1}`);
    });

    test('solve is deterministic and carries no hidden state across calls (frozen reuse)', () => {
        const { fx, C, cells } = setup(240, DEFAULT_THETA, 'barylen');
        const solver = prepareSchur(fx.op, fx.P, C);
        const first = cells.map((rhs) => solver.solve(rhs.top, rhs.bottom));
        for (const rhs of [...cells].reverse()) solver.solve(rhs.top, rhs.bottom);
        const again = cells.map((rhs) => solver.solve(rhs.top, rhs.bottom));
        for (let i = 0; i < cells.length; i++) {
            expect(again[i].x.every((v, j) => Object.is(v, first[i].x[j]))).toBe(true);
            expect(again[i].lambda.every((v, j) => Object.is(v, first[i].lambda[j]))).toBe(true);
        }
        // Omitting rhsBottom is exactly zeros(k), `solveSaddleFromA`'s default.
        const r0 = solver.solve(cells[0].top);
        const r1 = solver.solve(cells[0].top, zeros(C.length));
        expect(r0.x.every((v, j) => Object.is(v, r1.x[j]))).toBe(true);
    });

    test('k = 0 is a first-class case: no S, no W, solve is one GMRES', () => {
        const { fx, C, cells, k } = setup(240, DEFAULT_THETA, 'k0');
        expect(k).toBe(0);
        const { c, P } = counting(fx.P);
        const solver = prepareSchur(fx.op, P, C);
        expect(c.applies).toBe(0);
        expect(solver.report.columns).toEqual([]);
        expect(solver.report.sConditioning).toBeNull();
        expect(solver.report.tConditioning).toBeNull();
        expect(solver.report.pinned).toBe(false);
        const r = solver.solve(cells[0].top);
        expect(c.applies).toBe(r.gmres.iters + 1);
        expect(r.lambda).toEqual([]);
        // The seam passes rhsBottom = [] explicitly (`solveSaddleFrozen` fills zeros(k)).
        const r2 = solver.solve(cells[0].top, []);
        expect(r2.x.every((v, j) => Object.is(v, r.x[j]))).toBe(true);
        expect(() => solver.solve(cells[0].top, [1])).toThrow(RangeError);
    });
});

// ===========================================================================
// 5. The translation gauge (k = 0 and totalLength-only), documented in schur.ts.
// ===========================================================================

describe('translation gauge where the dense saddle is singular (k = 0, totalLength alone)', () => {
    for (const name of ['k0', 'len'] as const) {
        test(`n=240 ${name}: Nᵀx = Nᵀb/σ; the returned residual certifies consistency`, () => {
            const { fx, C, cells } = setup(240, DEFAULT_THETA, name);
            const n = 240;
            const solver = prepareSchur(fx.op, fx.P, C);
            // (a) a mean-free rhs (an energy gradient): consistent, residual at the GMRES level.
            const grad = solver.solve(cells[0].top, cells[0].bottom);
            expect(grad.residual).toBeLessThan(1e-6);
            // (b) a rhs WITH a mean: Ā x = b has no solution (Ā 1 = 0), and the gauge is Nᵀx = Nᵀb/σ.
            const b = lcgVec(3 * n, 7).map((v) => v + 0.3);
            const r = solver.solve(b, name === 'k0' ? undefined : [0.01]);
            for (let c = 0; c < 3; c++) {
                let sb = 0;
                let sx = 0;
                for (let i = 0; i < n; i++) {
                    sb += b[c * n + i];
                    sx += r.x[c * n + i];
                }
                expect(Math.abs(sx - sb / fx.op.sigma)).toBeLessThan(
                    1e-9 * Math.abs(sb / fx.op.sigma),
                );
            }
            // The unshifted saddle residual sees the inconsistency: 3 blocks × Σb/√n is O(1) here.
            expect(r.residual).toBeGreaterThan(1e-2);
            log(`${name} gauge residual (mean 0.3 rhs)`, r.residual.toExponential(2));
        });
    }
});

// ===========================================================================
// 6. Typed errors: route, rank, convergence.
// ===========================================================================

describe('typed errors (spec §8.1, §8.3, A33)', () => {
    test('edgeLengthsBlock (k = O(n)) is REFUSED BY NAME at n = 960 and at n = 24, before any GMRES', () => {
        for (const n of [960, 24]) {
            const fx = fixture(n, DEFAULT_THETA);
            const set = [edgeLengthsBlock(edgeLengths(fx.vertices, fx.edges))];
            const { C } = evaluateConstraintSet(set, fx.vertices, fx.edges);
            expect(C.length).toBe(n); // |E| = |V| on a closed ring: k = O(n) exactly, spec §8.1
            const { c, P } = counting(fx.P);
            let caught: unknown;
            try {
                prepareSchur(fx.op, P, C);
            } catch (e) {
                caught = e;
            }
            expect(caught).toBeInstanceOf(SchurRouteError);
            const err = caught as SchurRouteError;
            expect(err.k).toBe(n);
            expect(err.maxK).toBe(SCHUR_MAX_K);
            expect(err.message).toContain('edgeLengthsBlock');
            expect(err.message).toContain('nullspace');
            expect(err.message).toContain('§8.1');
            // Not a SINGULAR matrix: it must escape the seam's 'singular_system' fold
            // (optimizer.ts:448), which now tests the NEUTRAL base (spec A35 item 6, row 8b).
            expect(err).toBeInstanceOf(SolveNotCertifiedError);
            // ...and specifically NOT the EXTERNAL arm: `instanceof ExternalSolveError` is what a
            // GPU "re-run on the CPU path" recovery keys on, and re-running this CPU GMRES would
            // fail identically. @see src/core/sobolev/linsolve.ts (SolveNotCertifiedError)
            expect(err).not.toBeInstanceOf(ExternalSolveError);
            expect(c.applies).toBe(0);
        }
    });

    test('maxK is a real knob: k = 4 is refused at maxK = 3 and accepted at maxK = 4', () => {
        const { fx, C } = setup(48, DEFAULT_THETA, 'barylen');
        expect(C.length).toBe(4);
        expect(() => prepareSchur(fx.op, fx.P, C, { maxK: 3 })).toThrow(SchurRouteError);
        expect(prepareSchur(fx.op, fx.P, C, { maxK: 4 }).report.k).toBe(4);
        // The default cap: 3 (barycentre) + 1 (length) + 4 pins × 3 = 16.
        expect(SCHUR_MAX_K).toBe(16);
        const rows = Array.from({ length: SCHUR_MAX_K + 1 }, (_, i) => {
            const row = zeros(3 * 48);
            row[i] = 1;
            return row;
        });
        expect(() => prepareSchur(fx.op, fx.P, rows)).toThrow(SchurRouteError);
    });

    test('rank-deficient C is a SchurRankError (a plain Error: a genuinely singular system)', () => {
        const fx = fixture(48, DEFAULT_THETA);
        const pin = (v: number) => {
            const row = zeros(3 * 48);
            row[v] = 1;
            return row;
        };
        // (a) a duplicated pin: two identical rows ⇒ identical W columns ⇒ S has a ~1e-16 pivot.
        const dup = evaluateConstraintSet(
            [pointBlock(5, [0, 0, 0]), pointBlock(5, [0, 0, 0])],
            fx.vertices,
            fx.edges,
        ).C;
        expect(dup.length).toBe(6);
        let e1: unknown;
        try {
            prepareSchur(fx.op, fx.P, dup);
        } catch (e) {
            e1 = e;
        }
        expect(e1).toBeInstanceOf(SchurRankError);
        expect((e1 as SchurRankError).matrix).toBe('S');
        // A rank-deficient C really IS a singular system, so it must NOT escape the fold -- which
        // now means not being a `SolveNotCertifiedError` either (spec A35 item 6, row 8b).
        expect(e1).not.toBeInstanceOf(ExternalSolveError);
        expect(e1).not.toBeInstanceOf(SolveNotCertifiedError);
        // (b) a zero row.
        let e2: unknown;
        try {
            prepareSchur(fx.op, fx.P, [pin(5), zeros(3 * 48)]);
        } catch (e) {
            e2 = e;
        }
        expect(e2).toBeInstanceOf(SchurRankError);
        expect((e2 as SchurRankError).matrix).toBe('C');
        // (c) a PARTIAL translation pin: one row pins x-translation only, G has rank 1.
        let e3: unknown;
        try {
            prepareSchur(fx.op, fx.P, [pin(5)]);
        } catch (e) {
            e3 = e;
        }
        expect(e3).toBeInstanceOf(SchurRankError);
        expect((e3 as SchurRankError).matrix).toBe('T');
    });

    test('GMRES non-convergence is a SchurConvergenceError, NOT a singular system (trap 4)', () => {
        const { fx, C, cells } = setup(240, DEFAULT_THETA, 'pin');
        // maxit = 3 cannot reach η = 1e-10 at n = 240 (it takes ~45 iterations).
        let caught: unknown;
        try {
            prepareSchur(fx.op, fx.P, C, { maxit: 3 });
        } catch (e) {
            caught = e;
        }
        expect(caught).toBeInstanceOf(SchurConvergenceError);
        const err = caught as SchurConvergenceError;
        expect(err.phase).toBe('prepare');
        expect(err.column).toBe(0);
        expect(err.record.stop).toBe('maxit');
        expect(err.record.censored).toBe(true);
        expect(err.record.eta).toBeGreaterThan(DEFAULT_ETA_ACCEPT);
        // The seam only lets a SolveNotCertifiedError through (optimizer.ts:448,
        // lineSearch.ts:329) -- widened from ExternalSolveError by row 8b (spec A35 item 6), and
        // this error must be on the NON-external side of that split.
        expect(err).toBeInstanceOf(SolveNotCertifiedError);
        expect(err).not.toBeInstanceOf(ExternalSolveError);
        expect(err).not.toBeInstanceOf(SchurRankError);
        expect(err.message).toContain('NOT a singular system');
        // The same budget with the certificate off does NOT throw -- so it was the η check that fired.
        const lax = prepareSchur(fx.op, fx.P, C, { maxit: 3, etaAccept: Number.POSITIVE_INFINITY });
        expect(lax.report.columns.every((r) => r.censored)).toBe(true);
        // And in solve(): a solver prepared fine, then a rhs that needs more than maxit iterations.
        const k0 = setup(240, DEFAULT_THETA, 'k0');
        let caught2: unknown;
        try {
            prepareSchur(k0.fx.op, k0.fx.P, k0.C, { maxit: 3 }).solve(k0.cells[0].top);
        } catch (e) {
            caught2 = e;
        }
        expect(caught2).toBeInstanceOf(SchurConvergenceError);
        expect((caught2 as SchurConvergenceError).phase).toBe('solve');
        expect((caught2 as SchurConvergenceError).column).toBe(-1);
        expect(cells.length).toBeGreaterThan(0);
    });

    test('guards: defect needs knockOut; shapes; σ consistency', () => {
        const { fx, C } = setup(48, DEFAULT_THETA, 'pin');
        expect(() => prepareSchur(fx.op, fx.P, C, { defect: 'naive-shift' })).toThrow(/knockOut/);
        expect(() => prepareSchur(fx.op, fx.P, [zeros(5)])).toThrow(RangeError);
        const solver = prepareSchur(fx.op, fx.P, C);
        expect(() => solver.solve(zeros(10))).toThrow(RangeError);
        expect(() => solver.solve(zeros(3 * 48), [1, 2])).toThrow(RangeError);
        // A preconditioner built with another σ than the operator's would only show as slow GMRES.
        const other = fx.withSigmaScale(2);
        expect(() => prepareSchur(fx.op, other.P, C)).toThrow(/sigma/);
    });

    test('the solver IS a HierarchicalSaddleSolver: it drops into solveSaddleFrozen unchanged', () => {
        const { fx, C, cells } = setup(240, DEFAULT_THETA, 'barylen');
        const solver = prepareSchur(fx.op, fx.P, C);
        // Compiles ⇒ structurally a `HierarchicalSaddleSolver`; runs ⇒ the seam's own shape guards pass.
        const viaSeam = solveSaddleFrozen(
            { kind: 'hierarchical', n: 240, C, op: fx.op, solver },
            cells[0].top,
            cells[0].bottom,
        );
        const direct = solver.solve(cells[0].top, cells[0].bottom);
        expect(viaSeam.x.every((v, j) => Object.is(v, direct.x[j]))).toBe(true);
    });
});

// ===========================================================================
// 7. S, T and the translation test, against independent dense arithmetic.
// ===========================================================================

describe('S = C A_σ⁻¹ Cᵀ and its conditioning, against dense arithmetic (n = 48)', () => {
    for (const name of ['bary', 'pin', 'barylen', 'barypin'] as const) {
        test(`${name}: report.S equals the dense Schur complement; λ_min/λ_max equal the Sturm eigenvalues`, () => {
            const n = 48;
            const { fx, C, k } = setup(n, DEFAULT_THETA, name);
            const solver = prepareSchur(fx.op, fx.P, C);
            // Independent dense A_σ = A_bct + (σ/n) 1 1ᵀ (scalar), inverted by a plain Cholesky.
            const As = Float64Array.from(fx.denseOp());
            for (let i = 0; i < n * n; i++) As[i] += fx.op.sigma / n;
            const L = cholesky(As, n);
            const apply = (row: number[]): number[] => {
                const out = new Array<number>(3 * n);
                for (let c = 0; c < 3; c++) {
                    const v = Float64Array.from(row.slice(c * n, (c + 1) * n));
                    cholSolve(L, n, v);
                    for (let i = 0; i < n; i++) out[c * n + i] = v[i];
                }
                return out;
            };
            const W = C.map(apply);
            const Sref = new Float64Array(k * k);
            for (let r = 0; r < k; r++) {
                for (let s = 0; s < k; s++) {
                    let d = 0;
                    for (let i = 0; i < 3 * n; i++) d += C[r][i] * W[s][i];
                    Sref[r * k + s] = d;
                }
            }
            let dmax = 0;
            let smax = 0;
            for (let i = 0; i < k * k; i++) {
                dmax = Math.max(dmax, Math.abs(solver.report.S[i] - Sref[i]));
                smax = Math.max(smax, Math.abs(Sref[i]));
            }
            log(`${name} |S - S_dense|/|S|`, (dmax / smax).toExponential(2));
            expect(dmax / smax).toBeLessThan(1e-9);
            // SPD: the symmetric defect is the GMRES error in W, tiny at the default tol.
            expect(solver.report.sDefect).toBeLessThan(1e-9);
            const sp = spectrum(Float64Array.from(solver.report.S), k);
            const cond = solver.report.sConditioning;
            if (!cond) throw new Error('no conditioning');
            expect(sp.lambda1).toBeGreaterThan(0);
            expect(Math.abs(cond.lambdaMin - sp.lambda1) / sp.lambda1).toBeLessThan(1e-8);
            expect(Math.abs(cond.lambdaMax - sp.lambdaMax) / sp.lambdaMax).toBeLessThan(1e-8);
            expect(cond.cond).toBeCloseTo(cond.lambdaMax / cond.lambdaMin, 12);
        });
    }

    test('bary: G = C N = L·I exactly, so T = L² S⁻¹ and λ_min(T) = L²/λ_max(S)', () => {
        const n = 48;
        const { fx, C } = setup(n, DEFAULT_THETA, 'bary');
        const solver = prepareSchur(fx.op, fx.P, C);
        const L = totalLen(fx.vertices, fx.edges);
        for (let r = 0; r < 3; r++) {
            for (let c = 0; c < 3; c++) {
                let g = 0;
                for (let i = 0; i < n; i++) g += C[r][c * n + i];
                expect(Math.abs(g - (r === c ? L : 0))).toBeLessThan(1e-12 * L);
            }
        }
        const s = solver.report.sConditioning;
        const t = solver.report.tConditioning;
        if (!s || !t) throw new Error('no conditioning');
        expect(Math.abs(t.lambdaMin - (L * L) / s.lambdaMax) / t.lambdaMin).toBeLessThan(1e-8);
        expect(Math.abs(t.lambdaMax - (L * L) / s.lambdaMin) / t.lambdaMax).toBeLessThan(1e-8);
    });

    test('the translation-invariance cut sits in the middle of a >14-decade gap', () => {
        const sens = (name: SetName) => {
            const { fx, C } = setup(48, DEFAULT_THETA, name);
            return prepareSchur(fx.op, fx.P, C).report.translationSensitivity;
        };
        const len = sens('len');
        const pinning = Math.min(sens('bary'), sens('pin'), sens('barypin'));
        log(
            'translationSensitivity len / min(pinning sets)',
            `${len.toExponential(2)} / ${pinning.toExponential(2)}`,
        );
        expect(len).toBeLessThan(1e-14);
        expect(pinning).toBeGreaterThan(1e-2);
        expect(sens('k0')).toBe(0);
        // The cut must be INSIDE the gap it was chosen for, with >=3 decades of air on each side.
        // Asserted through the exported constant: the two literals above pin the MEASUREMENT, this
        // pins the THRESHOLD, and without it 1e-10 could drift anywhere in 1e-14..1e-2 green.
        // @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md A35
        expect(TRANSLATION_INVARIANT_REL).toBeGreaterThan(len * 1e3);
        expect(TRANSLATION_INVARIANT_REL).toBeLessThan(pinning / 1e3);
    });
});

// ===========================================================================
// 8. The knock-out table: which defect trips which cell. A gate that cannot be tripped reads like
//    one that passed (CLAUDE.md), so this table is the evidence the cells above are not vacuous.
// ===========================================================================

describe('knock-outs: each injected defect is caught by the cells the table says (n = 240)', () => {
    const TRIP = 1e-5; // 1e5 × the bar: every "caught" defect below is >= 1.5e-3
    /**
     * err vs dense LDLᵀ per cell of one set, for one (defect, tol), memoised on all three.
     * Why: `table(defect)` and the assertions below asked for the same tuples again (55 calls, 26
     * distinct), each one a fresh `prepareSchur` + GMRES solves; everything here is deterministic,
     * so a repeat returned the same numbers. The key MUST include `tol`: `errs('barypin',
     * undefined, 1e-6)` is a different solver from the default-tol one, and the 'no-symmetrise'
     * test asserts the two differ. Callers only read the map (test-suite review 2026-11-24).
     */
    const errsCache = new Map<string, ReadonlyMap<string, number>>();
    function errs(name: SetName, defect?: SchurDefect, tol?: number): ReadonlyMap<string, number> {
        const key = `${name}/${defect ?? 'clean'}/${tol ?? 'default'}`;
        const hit = errsCache.get(key);
        if (hit) return hit;
        const out = errsUncached(name, defect, tol);
        errsCache.set(key, out);
        return out;
    }
    function errsUncached(name: SetName, defect?: SchurDefect, tol?: number): Map<string, number> {
        const { fx, C, cells } = setup(240, DEFAULT_THETA, name);
        const refs = denseRefs(fx, DEFAULT_THETA, 'op', name, C, cells);
        const solver = prepareSchur(fx.op, fx.P, C, {
            ...(defect ? { knockOut: true, defect } : {}),
            ...(tol === undefined ? {} : { tol, etaAccept: Number.POSITIVE_INFINITY }),
        });
        const out = new Map<string, number>();
        for (const rhs of cells) {
            const r = solver.solve(rhs.top, rhs.bottom);
            const ref = refs.get(rhs.name);
            if (!ref) throw new Error('missing ref');
            out.set(
                rhs.name,
                UNPINNED.has(name)
                    ? relErr(demean(r.x, 240), demean(ref.ldltX, 240))
                    : relErr(r.x, ref.ldltX),
            );
        }
        return out;
    }
    const table = (defect: SchurDefect) => {
        const rows: string[] = [];
        for (const name of SET_NAMES) {
            const e = errs(name, defect);
            rows.push(`${name}: ${[...e].map(([c, v]) => `${c}=${v.toExponential(1)}`).join(' ')}`);
        }
        log(`knock-out '${defect}'`, `\n  ${rows.join('\n  ')}`);
    };

    test("'naive-shift' (S λ = C y − d with A_σ: the Ā vs A_σ trap) is caught on every pinning set", () => {
        table('naive-shift');
        for (const name of ['bary', 'barylen', 'pin', 'barypin'] as const) {
            for (const [cell, v] of errs(name, 'naive-shift')) {
                expect(v, `${name} ${cell}`).toBeGreaterThan(TRIP);
            }
        }
        // BLIND by construction where nothing pins the translations: G = 0, the shift is inert and
        // the naive formula IS the exact one. (Not a gap in the gate -- there is nothing to trip.)
        for (const name of ['k0', 'len'] as const) {
            const clean = errs(name);
            for (const [cell, v] of errs(name, 'naive-shift')) {
                expect(Math.abs(v - (clean.get(cell) ?? 0))).toBeLessThan(BAR);
            }
        }
    });

    test("'lambda-sign' is caught wherever λ ≠ 0 -- and NOT on k = 3 gradient/projection cells", () => {
        table('lambda-sign');
        const caught: [SetName, string][] = [
            ['bary', 'gen/-phi'],
            ['pin', 'gen/-phi'],
            ['barylen', 'grad/0'],
            ['barylen', 'zero/-phi'],
            ['barylen', 'gen/-phi'],
            ['barypin', 'grad/0'],
            ['barypin', 'zero/-phi'],
            ['barypin', 'gen/-phi'],
            ['len', 'grad/0'],
            ['len', 'zero/-phi'],
            ['len', 'gen/-phi'],
        ];
        for (const [name, cell] of caught) {
            expect(errs(name, 'lambda-sign').get(cell), `${name} ${cell}`).toBeGreaterThan(TRIP);
        }
        // THE FINDING: with G = C N square and invertible (bary, pin) the solvability condition
        // Gᵀλ = Nᵀb fixes λ ≈ 0 for any mean-free top, so a flipped λ sign is INVISIBLE on grad/0 and
        // zero/-phi (MEASURED 2.1e-12 / 5.6e-13 on bary). `gen/-phi` is the cell that exists to catch it.
        for (const name of ['bary', 'pin'] as const) {
            const clean = errs(name);
            const bad = errs(name, 'lambda-sign');
            for (const cell of ['grad/0', 'zero/-phi']) {
                expect(
                    Math.abs((bad.get(cell) ?? 0) - (clean.get(cell) ?? 0)),
                    `${name} ${cell}`,
                ).toBeLessThan(BAR);
            }
        }
    });

    test("'no-symmetrise' is NOT caught by the 1e-10 bar -- its effect is bounded by sDefect, and exists", () => {
        table('no-symmetrise');
        // THE FINDING: at the shipped tol, S's asymmetry is ~1e-13 (the GMRES error in W), so factoring
        // the lower triangle instead of the average moves x by far less than the bar. This knock-out
        // PASSES every cell above: the symmetrisation is hygiene, not a correctness gate.
        for (const name of SET_NAMES) {
            const clean = errs(name);
            for (const [cell, v] of errs(name, 'no-symmetrise')) {
                expect(Math.abs(v - (clean.get(cell) ?? 0)), `${name} ${cell}`).toBeLessThan(BAR);
            }
        }
        // It is live code, not dead code: with W solved loosely (tol 1e-6, sDefect ~1e-6) the two
        // variants differ and the symmetrised one is the closer to the dense answer, on every cell.
        const clean = errs('barypin', undefined, 1e-6);
        const bad = errs('barypin', 'no-symmetrise', 1e-6);
        for (const [cell, v] of bad) {
            log(
                `no-symmetrise vs clean at tol 1e-6, barypin ${cell}`,
                `${v.toExponential(2)} vs ${(clean.get(cell) ?? 0).toExponential(2)}`,
            );
            expect(v).toBeGreaterThan(1.1 * (clean.get(cell) ?? 0));
        }
    });

    test('a solver prepared for a STALE C (W reused across a changed C) is caught', () => {
        const n = 240;
        const fx = fixture(n, DEFAULT_THETA);
        const x0 = barycenterTarget(fx.vertices, fx.edges);
        const L0 = 0.97 * totalLen(fx.vertices, fx.edges);
        const mk = (dx: number) =>
            evaluateConstraintSet(
                [barycenterBlock([x0[0] + dx, x0[1] - 0.03, x0[2] + 0.02]), totalLengthBlock(L0)],
                fx.vertices,
                fx.edges,
            );
        const fresh = mk(0.05);
        const moved = mk(0.4); // a different barycentre target ⇒ a different C (rvec = m_I − x₀ enters dΦ)
        expect(dist(fresh.C.flat(), moved.C.flat())).toBeGreaterThan(1e-3);
        const top = lcgVec(3 * n, 11);
        const A = fx.denseOp();
        const ref = solveSaddleFromA(
            A,
            n,
            moved.C,
            top,
            moved.phi.map((v) => -v),
        );
        const staleSolver = prepareSchur(fx.op, fx.P, fresh.C);
        const rightSolver = prepareSchur(fx.op, fx.P, moved.C);
        const stale = relErr(
            staleSolver.solve(
                top,
                moved.phi.map((v) => -v),
            ).x,
            ref.x,
        );
        const right = relErr(
            rightSolver.solve(
                top,
                moved.phi.map((v) => -v),
            ).x,
            ref.x,
        );
        log('stale-C vs fresh-C error', `${stale.toExponential(2)} vs ${right.toExponential(2)}`);
        expect(right).toBeLessThan(BAR);
        expect(stale).toBeGreaterThan(TRIP);
    });

    test('the bottom block is live: negating rhsBottom (C x = −d) is caught by every pinning set', () => {
        for (const name of ['bary', 'barylen', 'pin', 'barypin'] as const) {
            const { fx, C, cells } = setup(240, DEFAULT_THETA, name);
            const refs = denseRefs(fx, DEFAULT_THETA, 'op', name, C, cells);
            const solver = prepareSchur(fx.op, fx.P, C);
            const rhs = cells.find((c) => c.name === 'zero/-phi');
            const ref = refs.get('zero/-phi');
            if (!rhs?.bottom || !ref) throw new Error('missing cell');
            const flipped = solver.solve(
                rhs.top,
                rhs.bottom.map((v) => -v),
            );
            const e = relErr(flipped.x, ref.ldltX);
            log(`${name} zero/-phi with rhsBottom negated`, e.toExponential(2));
            expect(e).toBeGreaterThan(TRIP);
        }
    });

    test("the returned residual also flags 'naive-shift' (self-certification is not blind to the trap)", () => {
        const { fx, C, cells } = setup(240, DEFAULT_THETA, 'pin');
        const bad = prepareSchur(fx.op, fx.P, C, { knockOut: true, defect: 'naive-shift' });
        const good = prepareSchur(fx.op, fx.P, C);
        const r = cells[0];
        const rb = bad.solve(r.top, r.bottom).residual;
        const rg = good.solve(r.top, r.bottom).residual;
        log(
            'pin grad/0 residual naive-shift vs exact',
            `${rb.toExponential(2)} vs ${rg.toExponential(2)}`,
        );
        expect(rg).toBeLessThan(1e-6);
        expect(rb).toBeGreaterThan(1e-3);
    });
});

// ===========================================================================
// 9. The pivot floor, bracketed by its own data. Section 7's rank cells prove RANK_PIVOT_REL is not
//    too SMALL (at 1e-20 the duplicate pin's ~1e-16 pivot sails through and (a) fails). Nothing
//    proved it is not too LARGE: at 1e-2 every cell in this file is still green, because the
//    best-conditioned legitimate S only reaches 1/cond ~ 1.7e-2. That is 14 decades of silent
//    drift in a guard, so it is bracketed here from both sides.
//    @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md A35
// ===========================================================================
describe('schur: the rank floor sits strictly inside the gap it was chosen for', () => {
    test('RANK_PIVOT_REL is >=3 decades under every legitimate 1/cond(S) and >=3 over the degenerate pivot', () => {
        // Upper side: a pivot of a well-conditioned S is bounded below by λ_min/S_jj >= 1/cond(S).
        // If the floor ever rose to meet that, a VALID constraint set would be rejected as singular.
        let worst = Number.POSITIVE_INFINITY;
        for (const name of ['bary', 'len', 'barylen', 'pin', 'barypin'] as const) {
            const { fx, C } = setup(48, DEFAULT_THETA, name);
            const cond = prepareSchur(fx.op, fx.P, C).report.sConditioning;
            if (!cond) throw new Error(`no conditioning for ${name}`);
            worst = Math.min(worst, 1 / cond.cond);
        }
        log('min legitimate 1/cond(S)', worst.toExponential(2));
        expect(worst).toBeGreaterThan(1e-3);
        expect(RANK_PIVOT_REL).toBeLessThan(worst / 1e3);

        // Lower side. NOT measurable from section 7's duplicated pin: two IDENTICAL rows give
        // identical `W` columns, so the eliminated pivot is EXACTLY 0.0 (MEASURED) and a floor of
        // 1e-300 would trip that cell too -- i.e. the duplicate cell constrains this constant only
        // to "> 0", which is no constraint at all. So the lower side is measured off a NEARLY
        // dependent pair instead, whose pivot is tunable: rows `e_v` and `e_v + delta*e_w` are
        // independent for delta > 0, and eliminating the first leaves a pivot of order delta^2.
        // Sweeping delta turns the floor's position into a measured trip/no-trip boundary.
        const fx = fixture(48, DEFAULT_THETA);
        const nearDup = (delta: number): number[][] => {
            const rows: number[][] = [];
            for (let c = 0; c < 3; c++) {
                const r = zeros(3 * 48);
                r[c * 48 + 5] = 1; // pins all three translations on its own (rank(C N) = 3)
                rows.push(r);
            }
            for (let c = 0; c < 3; c++) {
                const r = zeros(3 * 48);
                r[c * 48 + 5] = 1;
                r[c * 48 + 9] = delta;
                rows.push(r);
            }
            return rows;
        };
        const pivotOf = (delta: number): number | 'ok' => {
            let err: unknown;
            try {
                const rep = prepareSchur(fx.op, fx.P, nearDup(delta)).report;
                if (!rep.sConditioning) throw new Error('no conditioning');
                return 'ok';
            } catch (e) {
                err = e;
            }
            expect(err).toBeInstanceOf(SchurRankError);
            expect((err as SchurRankError).matrix).toBe('S');
            // Read the pivot out of the message the floor itself raised, so the number asserted is
            // the one the guard actually compared against RANK_PIVOT_REL.
            const m = /p=([-\d.e+]+), diag=([-\d.e+]+)/.exec((err as Error).message);
            if (!m) throw new Error(`pivot not in message: ${(err as Error).message}`);
            return Math.abs(Number(m[1]) / Number(m[2]));
        };
        const loose = pivotOf(1e-3); // pivot ~ 1e-6: well above the floor, must be ACCEPTED
        const tight = pivotOf(1e-8); // pivot ~ 1e-16: below the floor, must be REJECTED
        log('near-dup delta=1e-3 / delta=1e-8', `${loose} / ${tight}`);
        expect(loose).toBe('ok');
        expect(typeof tight).toBe('number');
        expect(tight as number).toBeLessThan(RANK_PIVOT_REL);
        // And the boundary is the floor's, not an accident of the arithmetic: the rejected pivot is
        // non-zero, so lowering RANK_PIVOT_REL far enough WOULD let this rank-deficient set through.
        expect(tight as number).toBeGreaterThan(0);
    });
});
