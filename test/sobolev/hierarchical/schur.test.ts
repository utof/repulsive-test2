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
import { trefoil } from '../../../src/core/fixtures';
import { DEFAULTS } from '../../../src/core/optimizer';
import {
    barycenterBlock,
    type ConstraintSet,
    edgeLengths,
    edgeLengthsBlock,
    evaluateConstraintSet,
    pointBlock,
    totalLengthBlock,
} from '../../../src/core/sobolev/constraintSet';
import { barycenterTarget } from '../../../src/core/sobolev/constraints';
import { applyAScalar, prepareA } from '../../../src/core/sobolev/hierarchical/apply';
import { buildBct, DEFAULT_THETA } from '../../../src/core/sobolev/hierarchical/bct';
import { buildBvh } from '../../../src/core/sobolev/hierarchical/bvh';
import { prepareLowOrder } from '../../../src/core/sobolev/hierarchical/lowOrder';
import {
    type HierarchicalOperator,
    makeHierarchicalOperator,
} from '../../../src/core/sobolev/hierarchical/operator';
import {
    makePreconditioner,
    type Preconditioner,
} from '../../../src/core/sobolev/hierarchical/precond';
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
import { assembleAFlat } from '../../../src/core/sobolev/innerProduct';
import {
    ExternalSolveError,
    SolveNotCertifiedError,
    solveSaddleFromA,
    solveSaddleFrozen,
} from '../../../src/core/sobolev/linsolve';
import { calculateDisjointPairs, gradientAnalytical } from '../../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../../src/core/testConfigs';
import { fullDescribe, fullTest } from '../../tier';
import { cholesky, cholSolve, densify, spectrum } from './denseSpectrum';

setDefaultTimeout(900_000);

/** The core's DEFAULTS (`optimizer.ts:23`): `dispatch.ts` does not forward them, so a saddle built
 * outside `sobolevStepSet` must use these or the operator and the step disagree silently (A33). */
const { alpha: ALPHA, beta: BETA, epsilon: EPS } = DEFAULTS;
const LEAF = 8;
/** The 1e-10 bar of spec §11.2 row 9b. */
const BAR = 1e-10;

const log = (name: string, v: unknown) => console.log(`[schur] ${name} = ${v}`);

// ---------------------------------------------------------------------------
// Small vector helpers.
// ---------------------------------------------------------------------------

const nrm = (v: ArrayLike<number>): number => {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    return Math.sqrt(s);
};
const dist = (u: ArrayLike<number>, v: ArrayLike<number>): number => {
    let s = 0;
    for (let i = 0; i < u.length; i++) s += (u[i] - v[i]) ** 2;
    return Math.sqrt(s);
};
const relErr = (u: ArrayLike<number>, ref: ArrayLike<number>): number => dist(u, ref) / nrm(ref);
/** Removes each coordinate block's mean (the `Nᵀ` gauge). */
function demean(x: ArrayLike<number>, n: number): number[] {
    const out = Array.from({ length: 3 * n }, (_, i) => x[i]);
    for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += out[c * n + i];
        s /= n;
        for (let i = 0; i < n; i++) out[c * n + i] -= s;
    }
    return out;
}
function lcgVec(len: number, seed: number): number[] {
    let s = seed >>> 0;
    const out: number[] = [];
    for (let i = 0; i < len; i++) {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        out.push(s / 4294967296 - 0.5);
    }
    return out;
}
const zeros = (len: number): number[] => new Array<number>(len).fill(0);

// ---------------------------------------------------------------------------
// Double-double residual: the TRUTH instrument. Dekker/Knuth error-free transforms (JS has no fma).
// Why it exists: a dense f64 solve is accurate only to ~kappa*u (1e-10 at n = 960, measured), so it
// cannot referee a 1e-10 bar; one refinement step with a double-double residual can.
// ---------------------------------------------------------------------------

const SPLIT = 134217729; // 2^27 + 1
class DD {
    hi = 0;
    lo = 0;
    add(x: number): void {
        const s = this.hi + x;
        const bb = s - this.hi;
        this.lo += this.hi - (s - bb) + (x - bb);
        this.hi = s;
    }
    addProd(a: number, b: number): void {
        const p = a * b;
        const ca = SPLIT * a;
        const ah = ca - (ca - a);
        const al = a - ah;
        const cb = SPLIT * b;
        const bh = cb - (cb - b);
        const bl = b - bh;
        this.add(p);
        this.lo += ah * bh - p + ah * bl + al * bh + al * bl;
    }
    value(): number {
        return this.hi + this.lo;
    }
}
/** `rhs − K z` in double-double, `K = [[A⊗I₃, Cᵀ], [C, 0]]`, `A` n×n row-major. */
function ddResidual(
    A: Float64Array,
    n: number,
    C: number[][],
    z: number[],
    rhs: number[],
): number[] {
    const m = 3 * n;
    const k = C.length;
    const r = new Array<number>(m + k);
    for (let c = 0; c < 3; c++) {
        for (let i = 0; i < n; i++) {
            const acc = new DD();
            acc.add(rhs[c * n + i]);
            for (let j = 0; j < n; j++) acc.addProd(-A[i * n + j], z[c * n + j]);
            for (let q = 0; q < k; q++) acc.addProd(-C[q][c * n + i], z[m + q]);
            r[c * n + i] = acc.value();
        }
    }
    for (let q = 0; q < k; q++) {
        const acc = new DD();
        acc.add(rhs[m + q]);
        for (let j = 0; j < m; j++) acc.addProd(-C[q][j], z[j]);
        r[m + q] = acc.value();
    }
    return r;
}

// ---------------------------------------------------------------------------
// Fixtures: the hierarchical stack on trefoil(n), and its dense counterparts.
// ---------------------------------------------------------------------------

interface Fixture {
    n: number;
    vertices: Vec3[];
    edges: Edge[];
    op: HierarchicalOperator;
    P: Preconditioner;
    /** Rebuilds `{op, P}` with `σ` scaled -- for the σ-invariance test. */
    withSigmaScale(s: number): { op: HierarchicalOperator; P: Preconditioner };
    /** Dense matrix of the SAME operator (`densify(applyAScalar)`, symmetrised). Lazy. */
    denseOp(): Float64Array;
    /** `assembleAFlat`: the independent dense Ā. Lazy. */
    denseA(): Float64Array;
}
const fixtureCache = new Map<string, Fixture>();
/** `theta = 0` is the exact STREAMING operator (spec §7.1); otherwise the lists operator at `theta`. */
function fixture(n: number, theta: number): Fixture {
    const key = `${n}/${theta}`;
    const hit = fixtureCache.get(key);
    if (hit) return hit;
    const { vertices, edges } = trefoil(n);
    const bvh = buildBvh(vertices, edges, LEAF, EPS);
    const lists = {
        kind: 'lists',
        bct: buildBct(bvh, { theta: theta === 0 ? DEFAULT_THETA : theta }),
    } as const;
    const source = theta === 0 ? ({ kind: 'stream', params: { theta: 0 } } as const) : lists;
    const H = prepareA(bvh, n, source, { alpha: ALPHA, beta: BETA, epsilon: EPS });
    const op = makeHierarchicalOperator(H);
    // The preconditioner is built over the LISTS block source either way: `Ã⁻¹` is only an
    // approximate inverse, so it does not have to share the operator's accuracy.
    const Lo = prepareLowOrder(vertices, edges, bvh, lists, {
        alpha: ALPHA,
        beta: BETA,
        variant: 'vertex',
    });
    let dOp: Float64Array | undefined;
    let dA: Float64Array | undefined;
    const fx: Fixture = {
        n,
        vertices,
        edges,
        op,
        P: makePreconditioner(vertices, edges, Lo, op.sigma),
        withSigmaScale(s) {
            const op2: HierarchicalOperator = { ...op, sigma: op.sigma * s };
            return { op: op2, P: makePreconditioner(vertices, edges, Lo, op2.sigma) };
        },
        denseOp() {
            if (!dOp) {
                dOp = densify(n, (u, out) => applyAScalar(H, u, out));
                for (let i = 0; i < n; i++) {
                    for (let j = i + 1; j < n; j++) {
                        const a = 0.5 * (dOp[i * n + j] + dOp[j * n + i]);
                        dOp[i * n + j] = a;
                        dOp[j * n + i] = a;
                    }
                }
            }
            return dOp;
        },
        denseA() {
            dA ??= assembleAFlat(vertices, edges, calculateDisjointPairs(edges), ALPHA, BETA, EPS);
            return dA;
        },
    };
    fixtureCache.set(key, fx);
    return fx;
}

function totalLen(vertices: Vec3[], edges: Edge[]): number {
    let L = 0;
    for (const [a, b] of edges) {
        L += Math.hypot(
            vertices[a][0] - vertices[b][0],
            vertices[a][1] - vertices[b][1],
            vertices[a][2] - vertices[b][2],
        );
    }
    return L;
}

/**
 * The constraint sets under test. Targets are OFFSET from the current curve so that `Φ ≠ 0` and the
 * projection-style right-hand side `−Φ` is non-trivial (a barycentre block at its own barycentre has
 * `Φ = 0` and the "non-zero bottom" cell would silently be a zero-bottom cell).
 */
const SET_NAMES = ['k0', 'bary', 'barylen', 'len', 'pin', 'barypin'] as const;
type SetName = (typeof SET_NAMES)[number];
/** Sets whose rows do NOT pin the three translations: the dense saddle is singular there. */
const UNPINNED: ReadonlySet<SetName> = new Set(['k0', 'len']);
function constraintSet(name: SetName, vertices: Vec3[], edges: Edge[]): ConstraintSet {
    const x0 = barycenterTarget(vertices, edges);
    const x0off: Vec3 = [x0[0] + 0.05, x0[1] - 0.03, x0[2] + 0.02];
    const L0 = 0.97 * totalLen(vertices, edges);
    switch (name) {
        case 'k0':
            return [];
        case 'bary':
            return [barycenterBlock(x0off)];
        case 'barylen':
            return [barycenterBlock(x0off), totalLengthBlock(L0)];
        case 'len':
            return [totalLengthBlock(L0)];
        case 'pin':
            return [pointBlock(5, [0.1, 0.2, 0.3])];
        case 'barypin':
            return [barycenterBlock(x0off), pointBlock(7, [-0.2, 0.1, 0]), totalLengthBlock(L0)];
    }
}

interface Rhs {
    name: string;
    top: number[];
    /** `undefined` ⇒ omitted (zeros(k)), exactly `solveSaddleFromA`'s default. */
    bottom: number[] | undefined;
}
/**
 * Three right-hand sides, each there for a measured reason:
 *  - `grad/0`: the gradient solve (`dE`, bottom omitted). `dE` is translation-invariant so `Nᵀb ≈ 0`.
 *  - `zero/-phi`: the projection solve (`rhsTop = 0`, `rhsBottom = −Φ`), `lineSearch.ts:279`.
 *  - `gen/-phi`: a GENERIC top with non-zero per-block mean. It exists because with a k = 3 pinning
 *    set the solvability condition `Gᵀλ = Nᵀb` determines `λ` alone, so on the two cells above
 *    `λ ≈ 0` and a flipped-sign `λ` is INVISIBLE (MEASURED: knock-out 'lambda-sign' passes
 *    `grad/0` and `zero/-phi` on `bary` and `pin` at 2.1e-12 and 5.6e-13). Only this cell trips it.
 *    On sets that do not pin the translations the dense system is inconsistent for a non-zero-mean
 *    top, so the generic top is mean-free there.
 */
function rhsCells(fx: Fixture, name: SetName, k: number, phi: number[]): Rhs[] {
    const n = fx.n;
    const g = gradientAnalytical(
        fx.vertices,
        fx.edges,
        calculateDisjointPairs(fx.edges),
        ALPHA,
        BETA,
        EPS,
    );
    const grad = new Array<number>(3 * n);
    for (let c = 0; c < 3; c++) for (let i = 0; i < n; i++) grad[c * n + i] = g[i][c];
    const gen = lcgVec(3 * n, 99);
    const negPhi = phi.map((v) => -v);
    return [
        { name: 'grad/0', top: grad, bottom: undefined },
        ...(k === 0 ? [] : [{ name: 'zero/-phi', top: zeros(3 * n), bottom: negPhi }]),
        { name: 'gen/-phi', top: UNPINNED.has(name) ? demean(gen, n) : gen, bottom: negPhi },
    ];
}

/** One dense-reference answer for one right-hand side. */
interface CellRef {
    /** Refined truth `x` (pinned sets), or the LDLᵀ `x` where the dense saddle is singular. */
    x: number[];
    lambda: number[];
    /** The dense LDLᵀ answer, i.e. the lead-named oracle `solveSaddleFromA`. */
    ldltX: number[];
    ldltLambda: number[];
    /** Size of the LAST refinement correction relative to `‖z‖`; `NaN` when no truth exists. */
    lastCorrection: number;
}
const refCache = new Map<string, Map<string, CellRef>>();
/**
 * The dense reference of one `(fixture, matrix, set)`: ONE LDLᵀ factorization (`solveSaddleFromA`,
 * the oracle), reused for every right-hand side through `solveSaddleFrozen`, whose substitution is
 * the same `solveFactored` call. On pinned sets `x` is refined with a double-double residual and the
 * answer CERTIFIES itself (`lastCorrection` must be at roundoff). Cached because the n = 960
 * factorization costs ~8 s and three describes need it.
 */
function denseRefs(
    fx: Fixture,
    theta: number,
    matrix: 'op' | 'A',
    name: SetName,
    C: number[][],
    cells: Rhs[],
): Map<string, CellRef> {
    const key = `${fx.n}/${theta}/${matrix}/${name}`;
    const hit = refCache.get(key);
    if (hit) return hit;
    const n = fx.n;
    const m = 3 * n;
    const k = C.length;
    const A = matrix === 'op' ? fx.denseOp() : fx.denseA();
    const first = solveSaddleFromA(A, n, C, zeros(m), zeros(k));
    const frozen = { kind: 'dense' as const, a: A, n, C, fac: first.fac };
    const out = new Map<string, CellRef>();
    for (const rhs of cells) {
        const s0 = solveSaddleFrozen(frozen, rhs.top, rhs.bottom);
        if (UNPINNED.has(name)) {
            out.set(rhs.name, {
                x: s0.x,
                lambda: s0.lambda,
                ldltX: s0.x,
                ldltLambda: s0.lambda,
                lastCorrection: Number.NaN,
            });
            continue;
        }
        const full = [...rhs.top, ...(rhs.bottom ?? zeros(k))];
        let z = [...s0.x, ...s0.lambda];
        let last = Number.NaN;
        for (let it = 0; it < 2; it++) {
            const r = ddResidual(A, n, C, z, full);
            const dz = solveSaddleFrozen(frozen, r.slice(0, m), r.slice(m));
            const delta = [...dz.x, ...dz.lambda];
            last = nrm(delta) / nrm(z);
            z = z.map((v, i) => v + delta[i]);
        }
        out.set(rhs.name, {
            x: z.slice(0, m),
            lambda: z.slice(m),
            ldltX: s0.x,
            ldltLambda: s0.lambda,
            lastCorrection: last,
        });
    }
    refCache.set(key, out);
    return out;
}

/** `λ` agreement: absolute, scaled by `max(1, ‖λ_ref‖)` -- on k = 3 gradient cells `λ_ref ≈ 1e-12`. */
const lamErr = (lam: number[], ref: number[]): number => dist(lam, ref) / Math.max(1, nrm(ref));

/** Everything a cell needs, built once per `(n, theta, set)`. */
function setup(n: number, theta: number, name: SetName) {
    const fx = fixture(n, theta);
    const set = constraintSet(name, fx.vertices, fx.edges);
    const { C, phi } = evaluateConstraintSet(set, fx.vertices, fx.edges);
    return { fx, C, phi, k: C.length, cells: rhsCells(fx, name, C.length, phi) };
}

/**
 * Compares one solver against the dense reference over every rhs cell. `refBar` is the bar on the
 * agreement with the LITERAL LDLᵀ answer (asserted only where the dense error is far below it).
 */
function checkCells(
    n: number,
    theta: number,
    matrix: 'op' | 'A',
    name: SetName,
    opts: { literalBar: boolean; truthBar: boolean },
): void {
    const { fx, C, k, cells } = setup(n, theta, name);
    const solver = prepareSchur(fx.op, fx.P, C);
    const unpinned = UNPINNED.has(name);
    // Why: on an UNPINNED set with no literal bar (n = 960 k0 / len) no assertion reads the dense
    // reference -- the truth bar is skipped where the dense saddle is translation-singular -- so
    // building it (one ~8.5 s LDLᵀ of the 2880-dim saddle) only fed the log line. Skip it; the
    // log then shows `literal=NaN`. (test-suite review 2026-11-24, README.md "Tests")
    const needRef = opts.literalBar || (opts.truthBar && !unpinned);
    const refs = needRef ? denseRefs(fx, theta, matrix, name, C, cells) : null;
    expect(solver.report.pinned).toBe(!unpinned);
    // Structure: prepare ran exactly k GMRES, and ran them at the default tol.
    expect(solver.report.columns.length).toBe(k);
    expect(solver.report.tol).toBe(DEFAULT_SCHUR_TOL);
    for (const rhs of cells) {
        const r = solver.solve(rhs.top, rhs.bottom);
        const ref = refs?.get(rhs.name);
        if (refs && !ref) throw new Error(`no dense reference for ${rhs.name}`);
        // The literal oracle: the dense LDLᵀ x (mean-free where the dense system is singular, since
        // its translation is arbitrary: MEASURED, ldlt and lu return means 6.0e-2 vs 8.0e-2 at k = 0).
        const literal = !ref
            ? Number.NaN
            : unpinned
              ? relErr(demean(r.x, n), demean(ref.ldltX, n))
              : relErr(r.x, ref.ldltX);
        const lamLiteral = ref ? lamErr(r.lambda, ref.ldltLambda) : Number.NaN;
        const vsTruth = unpinned || !ref ? Number.NaN : relErr(r.x, ref.x);
        const denseVsTruth = unpinned || !ref ? Number.NaN : relErr(ref.ldltX, ref.x);
        const lamTruth = unpinned || !ref ? Number.NaN : lamErr(r.lambda, ref.lambda);
        log(
            `n=${n} th=${theta === 0 ? '0(stream)' : theta} ${name} ${rhs.name}`,
            `literal=${literal.toExponential(2)} vsTruth=${vsTruth.toExponential(2)} ` +
                `denseVsTruth=${denseVsTruth.toExponential(2)} lam(vsTruth)=${lamTruth.toExponential(2)} ` +
                `res=${r.residual.toExponential(2)} eta=${r.gmres.eta.toExponential(1)} ` +
                `it(prep)=${solver.report.columns.map((c) => c.iters).join('/') || '-'} it(solve)=${r.gmres.iters} ` +
                `S:[${solver.report.sConditioning?.lambdaMin.toExponential(2) ?? '-'},${solver.report.sConditioning?.lambdaMax.toExponential(2) ?? '-'}] ` +
                `condS=${solver.report.sConditioning?.cond.toExponential(2) ?? '-'} ` +
                `sDef=${solver.report.sDefect.toExponential(1)}`,
        );
        expect(r.x.length).toBe(3 * n);
        expect(r.lambda.length).toBe(k);
        // Self-certification: the returned residual is formed with the UNSHIFTED Ā and the real matvec.
        expect(r.residual).toBeLessThan(1e-6);
        expect(r.gmres.eta).toBeLessThan(DEFAULT_ETA_ACCEPT);
        if (opts.literalBar) {
            expect(literal).toBeLessThan(BAR);
            expect(lamLiteral).toBeLessThan(1e-9);
        }
        if (opts.truthBar && !unpinned) {
            if (!ref) throw new Error('truth bar without a dense reference'); // needRef guarantees it
            // The truth is only a referee if refinement converged to roundoff.
            expect(ref.lastCorrection).toBeLessThan(1e-14);
            expect(vsTruth).toBeLessThan(BAR);
            expect(lamTruth).toBeLessThan(1e-9);
        }
    }
}

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
    for (const name of SET_NAMES) {
        // n = 960: the literal agreement is LOGGED (it plateaus at the oracle's own error); the 1e-10
        // bar is asserted against the refined truth.
        // FULL TIER ONLY (test/tier.ts): one dense LDLᵀ + refinement of the 2880-dim saddle per set
        // (~8.5 s each). Why: the n = 240 cells above assert the literal AND the truth bar on every
        // set in the default tier; what waits for a full run is a defect that only shows at n = 960
        // (deeper tree, more far blocks, κ·u near the bar) (README.md "Tests").
        fullTest(`n=960 ${name}: 1e-10 vs refined truth (literal agreement logged)`, () => {
            checkCells(960, DEFAULT_THETA, 'op', name, { literalBar: false, truthBar: true });
        });
    }
});

describe('Schur vs dense: independent operator path (θ = 0 streaming vs assembleAFlat), n = 240', () => {
    // Why n = 240 only: a streaming apply is O(N²); at n = 960 preparing a k = 3 solver took 88 s and
    // one solve 22-33 s (MEASURED), so the suite cannot afford it. The n = 960 θ = 0 numbers are in
    // the task report: Schur is 1.7e-10 / 2.1e-10 from the Ā-truth there (the stream apply and
    // `assembleAFlat` differ by relFro 1.4e-15 at n = 480, ~12 ulp, and x moves 6-9e-12 per ulp-sized
    // random perturbation at n = 960).
    for (const name of SET_NAMES) {
        test(`n=240 ${name}: streaming operator vs assembleAFlat`, () => {
            checkCells(240, 0, 'A', name, { literalBar: true, truthBar: true });
        });
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

// FULL TIER ONLY (test/tier.ts): a calibration measurement, five prepareSchur + solves per set at
// n = 960 (~21 s). Why: it is the evidence behind DEFAULT_SCHUR_TOL (pinned by the default test
// above), not a per-merge check; re-run it (`bun run test:full`) whenever the tol, the GMRES
// stopping rule or the operator changes (README.md "Tests").
fullDescribe('tol sweep (spec §8.3; the knee behind DEFAULT_SCHUR_TOL), n = 960', () => {
    // MEASURED (n = 960, barylen and pin, grad/0 and gen/-phi), error vs refined truth:
    //   tol 1e-6: 2.5e-6..3.2e-6   1e-8: 3.3e-8..7.4e-8   1e-10: 2.2e-10..5.7e-10   1e-11: 2.1e-11..7.3e-11
    //   tol 1e-12: 3.7e-12..7.6e-12   1e-13: 2.2e-12..3.2e-12   1e-14: 2.2e-12..3.2e-12 (identical)
    // i.e. error ≈ 2-4 × the TRUE backward error η, with a floor of ~2-3e-12 (κ·u) that tol 1e-13
    // reaches and tol 1e-14 cannot improve. 1e-12 is within 2-3x of the floor, 13x under the bar, and
    // costs 46 iterations against 51 for 1e-13.
    for (const name of ['barylen', 'pin'] as const) {
        test(`${name}: error vs truth falls with tol to a floor; the chosen tol is within 4x of it`, () => {
            const { fx, C, cells } = setup(960, DEFAULT_THETA, name);
            const refs = denseRefs(fx, DEFAULT_THETA, 'op', name, C, cells);
            const errAt = (tol: number): { err: number; eta: number } => {
                // etaAccept: Infinity so a loose tol is observable instead of refused.
                const solver = prepareSchur(fx.op, fx.P, C, {
                    tol,
                    etaAccept: Number.POSITIVE_INFINITY,
                });
                let err = 0;
                let eta = 0;
                for (const rhs of cells) {
                    if (rhs.name === 'zero/-phi') continue; // no rhs GMRES: tests prepare only
                    const r = solver.solve(rhs.top, rhs.bottom);
                    const ref = refs.get(rhs.name);
                    if (!ref) throw new Error('missing ref');
                    err = Math.max(err, relErr(r.x, ref.x));
                    eta = Math.max(eta, r.gmres.eta, ...solver.report.columns.map((c) => c.eta));
                }
                return { err, eta };
            };
            const sweep = [1e-8, 1e-10, 1e-12, 1e-13, 1e-14].map((tol) => ({ tol, ...errAt(tol) }));
            for (const s of sweep) {
                log(
                    `tol sweep ${name}`,
                    `tol=${s.tol.toExponential(0)} err=${s.err.toExponential(2)} eta=${s.eta.toExponential(1)} err/eta=${(s.err / s.eta).toFixed(1)}`,
                );
            }
            const at = (t: number) => sweep.find((s) => s.tol === t) as (typeof sweep)[number];
            // Non-vacuity: a loose tol DOES violate the bar, so the truth bar above is a gate that can trip.
            expect(at(1e-8).err).toBeGreaterThan(BAR * 100);
            expect(at(1e-10).err).toBeGreaterThan(BAR);
            // The default clears the bar by >10x ...
            expect(DEFAULT_SCHUR_TOL).toBe(1e-12);
            expect(at(1e-12).err).toBeLessThan(BAR / 10);
            // ... and the knee: 1e-13 and 1e-14 are the same floor, 1e-12 is within 4x of it.
            const floor = at(1e-14).err;
            expect(at(1e-13).err / floor).toBeLessThan(1.5);
            expect(at(1e-12).err / floor).toBeLessThan(4);
            // Above the floor the forward error tracks the true backward error (the quantity etaAccept
            // certifies); below it η keeps falling while the error cannot, so the ratio is unbounded.
            for (const s of sweep.filter((q) => q.tol >= 1e-12))
                expect(s.err / s.eta).toBeLessThan(30);
        });
    }
});

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
