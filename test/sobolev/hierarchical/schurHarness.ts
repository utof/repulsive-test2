/**
 * The shared harness of `schur.test.ts` and `schur.n960.test.ts` (spec §8.2 / §11.2 row 9b): the
 * fixtures, the six constraint sets and their right-hand-side cells, the dense LDLᵀ references
 * refined by a double-double residual (the TRUTH instrument), and `checkCells`. Moved here verbatim
 * from `schur.test.ts`; read that file's module TSDoc for what the oracle can and cannot certify.
 *
 * Why a module: the test runner (`scripts/test.ts`) runs one process per file, and the full tier's
 * wall time was the schur file alone (431 s against 139 s for the next file, measured 2026-11-24
 * at load ~15), so its n = 960 group lives in its own file and both files need this harness. The
 * caches below (`fixtureCache`, `refCache`) are per process, i.e. per test file. Everything they
 * return is SHARED within a file: read-only.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.2, §8.3, §11.2 row 9b, A33
 * @issue utof/repulsive-test2#62
 */
import { expect } from 'bun:test';
import { trefoil } from '../../../src/core/fixtures';
import { DEFAULTS } from '../../../src/core/optimizer';
import {
    barycenterBlock,
    type ConstraintSet,
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
} from '../../../src/core/sobolev/hierarchical/schur';
import { assembleAFlat } from '../../../src/core/sobolev/innerProduct';
import { solveSaddleFromA, solveSaddleFrozen } from '../../../src/core/sobolev/linsolve';
import { calculateDisjointPairs, gradientAnalytical } from '../../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../../src/core/testConfigs';
import { densify } from './denseSpectrum';

/** The core's DEFAULTS (`optimizer.ts:23`): `dispatch.ts` does not forward them, so a saddle built
 * outside `sobolevStepSet` must use these or the operator and the step disagree silently (A33). */
const { alpha: ALPHA, beta: BETA, epsilon: EPS } = DEFAULTS;
const LEAF = 8;
/**
 * The 1e-10 bar of spec §11.2 row 9b.
 * @see test/sobolev/hierarchical/schur.test.ts (module TSDoc: what this oracle can and cannot certify)
 */
export const BAR = 1e-10;

/**
 * `[schur]`-prefixed log line, shared by both schur test files.
 * @see test/sobolev/hierarchical/schur.test.ts (module TSDoc: what this oracle can and cannot certify)
 */
export const log = (name: string, v: unknown) => console.log(`[schur] ${name} = ${v}`);

// ---------------------------------------------------------------------------
// Small vector helpers.
// ---------------------------------------------------------------------------

const nrm = (v: ArrayLike<number>): number => {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    return Math.sqrt(s);
};
/**
 * Euclidean distance `‖u − v‖₂`.
 * @see test/sobolev/hierarchical/schur.test.ts (module TSDoc: what this oracle can and cannot certify)
 */
export const dist = (u: ArrayLike<number>, v: ArrayLike<number>): number => {
    let s = 0;
    for (let i = 0; i < u.length; i++) s += (u[i] - v[i]) ** 2;
    return Math.sqrt(s);
};
/**
 * Relative error `‖u − ref‖ / ‖ref‖`.
 * @see test/sobolev/hierarchical/schur.test.ts (module TSDoc: what this oracle can and cannot certify)
 */
export const relErr = (u: ArrayLike<number>, ref: ArrayLike<number>): number =>
    dist(u, ref) / nrm(ref);
/**
 * Removes each coordinate block's mean (the `Nᵀ` gauge).
 * @see test/sobolev/hierarchical/schur.test.ts (module TSDoc: what this oracle can and cannot certify)
 */
export function demean(x: ArrayLike<number>, n: number): number[] {
    const out = Array.from({ length: 3 * n }, (_, i) => x[i]);
    for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += out[c * n + i];
        s /= n;
        for (let i = 0; i < n; i++) out[c * n + i] -= s;
    }
    return out;
}
/**
 * Deterministic LCG vector in `[-0.5, 0.5)` (no `Math.random`, so every cell is reproducible).
 * @see test/sobolev/hierarchical/schur.test.ts (module TSDoc: what this oracle can and cannot certify)
 */
export function lcgVec(len: number, seed: number): number[] {
    let s = seed >>> 0;
    const out: number[] = [];
    for (let i = 0; i < len; i++) {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        out.push(s / 4294967296 - 0.5);
    }
    return out;
}
/**
 * A zero vector of length `len`.
 * @see test/sobolev/hierarchical/schur.test.ts (module TSDoc: what this oracle can and cannot certify)
 */
export const zeros = (len: number): number[] => new Array<number>(len).fill(0);

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
/**
 * `theta = 0` is the exact STREAMING operator (spec §7.1); otherwise the lists operator at `theta`.
 * @see test/sobolev/hierarchical/schur.test.ts (module TSDoc: what this oracle can and cannot certify)
 */
export function fixture(n: number, theta: number): Fixture {
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

/**
 * Total edge length of a curve (the `totalLengthBlock` targets are offset from it).
 * @see test/sobolev/hierarchical/schur.test.ts (module TSDoc: what this oracle can and cannot certify)
 */
export function totalLen(vertices: Vec3[], edges: Edge[]): number {
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
 * @see test/sobolev/hierarchical/schur.test.ts (module TSDoc: what this oracle can and cannot certify)
 */
export const SET_NAMES = ['k0', 'bary', 'barylen', 'len', 'pin', 'barypin'] as const;
/**
 * One of `SET_NAMES`.
 * @see test/sobolev/hierarchical/schur.test.ts (module TSDoc: what this oracle can and cannot certify)
 */
export type SetName = (typeof SET_NAMES)[number];
/**
 * Sets whose rows do NOT pin the three translations: the dense saddle is singular there.
 * @see test/sobolev/hierarchical/schur.test.ts (module TSDoc: what this oracle can and cannot certify)
 */
export const UNPINNED: ReadonlySet<SetName> = new Set(['k0', 'len']);
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
 * @see test/sobolev/hierarchical/schur.test.ts (module TSDoc: what this oracle can and cannot certify)
 */
export function denseRefs(
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

/**
 * Everything a cell needs, built once per `(n, theta, set)`.
 * @see test/sobolev/hierarchical/schur.test.ts (module TSDoc: what this oracle can and cannot certify)
 */
export function setup(n: number, theta: number, name: SetName) {
    const fx = fixture(n, theta);
    const set = constraintSet(name, fx.vertices, fx.edges);
    const { C, phi } = evaluateConstraintSet(set, fx.vertices, fx.edges);
    return { fx, C, phi, k: C.length, cells: rhsCells(fx, name, C.length, phi) };
}

/**
 * Compares one solver against the dense reference over every rhs cell. `refBar` is the bar on the
 * agreement with the LITERAL LDLᵀ answer (asserted only where the dense error is far below it).
 * @see test/sobolev/hierarchical/schur.test.ts (module TSDoc: what this oracle can and cannot certify)
 */
export function checkCells(
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
