/**
 * Spec §8.2 / §11.2 row 9 of Phase 2H: `nullspace.ts`, the `k = O(n)` constraint route -- the `|E|`
 * `edgeLengths` rows eliminated by an implicit `O(n)` projection, composed with `schur.ts` for
 * whatever `O(1)` rows sit beside them.
 *
 * READING ORDER IS LOAD-BEARING. §8.3 item 2 requires the projected-preconditioned operator to be
 * verified SPD on the constrained subspace **before any convergence number is quoted**, so the SPD
 * block runs first and nothing before it asserts an iteration count, an `η` or a timing.
 *
 * WHAT EACH ORACLE CAN AND CANNOT CERTIFY (MEASURED -- this is why the cells carry several columns):
 *  1. **On a PINNED set** (`barycenterBlock` or `pointBlock` present) the dense saddle is
 *     non-singular and double-double iterative refinement converges in one step (`|δz|/|z|` 1.8e-13
 *     → 5e-17), so the bar is asserted against the REFINED TRUTH, per spec A35 item 1. Dense LDLᵀ's
 *     own error is 4.3e-13 at n = 240 and 1.4e-10…6.4e-10 at n = 960, so a literal agreement bar at
 *     n = 960 measures the oracle, not the solver.
 *  2. **On the PURE `edgeLengths` set there is NO truth instrument at all.** `C_edge N = 0` exactly,
 *     so the set is translation-singular, the dense saddle has `[N; 0]` in its kernel, and the DD
 *     refinement DIVERGES (MEASURED `|δz|/|z|` = 1.7e+0 → 2.0e+0 at n = 240). Dense LDLᵀ and LU
 *     disagree by 9.88e-1 there. Those cells are therefore asserted (a) MOD TRANSLATION against the
 *     dense LDLᵀ, and (b) against the route's OWN gauge identity `Nᵀ x = Nᵀ b / σ`, which is an
 *     identity rather than a reference. Spec A35 item 3 + A36 item 3.
 *  3. The BCT operator at `DEFAULT_THETA` is not `Ā` (θ compression moves `x` by 1.6e-3…4.5e-3,
 *     A35 item 8), so the oracle for the SOLVER is the dense matrix of the SAME operator. The
 *     independent-operator cell is the `nFar = 0` EXACT limit (A36 item 1: the only cell that can
 *     see a plumbing defect), at n = 240 only.
 *  4. `ℓ⁰` is PERTURBED in every cell. `edgeLengths(vertices, edges)` returns the CURRENT lengths,
 *     so an unperturbed target gives `Φ ≡ 0` identically and every cell would be degenerate -- A35
 *     item 5's corpus trap, in its sharpest form for this block.
 *
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §8.1, §8.2, §8.3, §11.2 row 9, A35, A36
 * @issue utof/repulsive-test2#62
 */
import { describe, expect, setDefaultTimeout, test } from 'bun:test';
import { trefoil } from '../../../src/core/fixtures';
import { DEFAULTS } from '../../../src/core/optimizer';
import {
    assertValidConstraintSet,
    barycenterBlock,
    type ConstraintBlock,
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
import { applyAsigma, gmres } from '../../../src/core/sobolev/hierarchical/gmres';
import { prepareLowOrder } from '../../../src/core/sobolev/hierarchical/lowOrder';
import {
    NullspaceRouteError,
    prepareNullspace,
} from '../../../src/core/sobolev/hierarchical/nullspace';
import {
    type HierarchicalOperator,
    makeHierarchicalOperator,
} from '../../../src/core/sobolev/hierarchical/operator';
import {
    makePreconditioner,
    type Preconditioner,
} from '../../../src/core/sobolev/hierarchical/precond';
import { DEFAULT_SCHUR_TOL, SCHUR_MAX_K } from '../../../src/core/sobolev/hierarchical/schur';
import { assembleAFlat } from '../../../src/core/sobolev/innerProduct';
import {
    operatorSaddleResidual,
    SolveNotCertifiedError,
    solveSaddleFromA,
    solveSaddleFrozen,
} from '../../../src/core/sobolev/linsolve';
import { calculateDisjointPairs, gradientAnalytical } from '../../../src/core/tangentPointEnergy';
import type { Edge, Vec3 } from '../../../src/core/testConfigs';
import { cholesky, densify, spectrum } from './denseSpectrum';

setDefaultTimeout(900_000);

const { alpha: ALPHA, beta: BETA, epsilon: EPS } = DEFAULTS;
const LEAF = 8;
/** Spec §11.2 row 9's bar, as amended by A35 item 1 (against the DD-refined truth). */
const BAR = 1e-10;

const log = (name: string, v: unknown) => console.log(`[nullspace] ${name} = ${v}`);

// ---------------------------------------------------------------------------
// Vector helpers.
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
/** Removes each coordinate block's mean -- the `Nᵀ` gauge (A35 item 3, A36 item 3). */
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
/** `Nᵀ x`, the three per-block sums. */
function blockSums(x: ArrayLike<number>, n: number): number[] {
    const out = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += x[c * n + i];
        out[c] = s;
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
// Double-double residual -- the TRUTH instrument (A35 item 1). Same Dekker/Knuth transforms as
// `schur.test.ts`; duplicated rather than imported because `schur.test.ts` is a test file, and a
// shared helper between two test files is a shared-failure mode (memory `common-mode-blindness`).
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

/**
 * `x` and `λ` of the dense saddle refined to `~5e-17` relative. Returns `null` with the divergence
 * trace when the refinement does NOT converge -- which is the honest answer on a translation-singular
 * set, and the fact the pure-`edgeLengths` cells rest on.
 *
 * Also returns `ldlt`, the UNREFINED `solveSaddleFromA(…, 'ldlt')` result the refinement starts
 * from. Why: a caller that needs the plain dense-LDLᵀ answer (the n = 960 "oracle over the bar"
 * test) reads it here instead of factoring the same 3843-dim saddle a second time -- same inputs,
 * same mode, deterministic, so the repeat was bit-identical and cost ~27 s (test-suite review
 * 2026-11-24, README.md "Tests").
 */
function refinedTruth(
    A: Float64Array,
    n: number,
    C: number[][],
    rhsTop: number[],
    rhsBottom: number[],
): {
    z: number[];
    trace: number[];
    converged: boolean;
    ldlt: { x: number[]; lambda: number[] };
} {
    const m = 3 * n;
    const ldlt = solveSaddleFromA(A, n, C, rhsTop, rhsBottom, 'ldlt');
    const rhs = [...rhsTop, ...rhsBottom];
    const frozen = { kind: 'dense' as const, a: A, n, C, fac: ldlt.fac };
    let z = [...ldlt.x, ...ldlt.lambda];
    const trace: number[] = [];
    for (let it = 0; it < 4; it++) {
        const r = ddResidual(A, n, C, z, rhs);
        const dz = solveSaddleFrozen(frozen, r.slice(0, m), r.slice(m));
        const delta = [...dz.x, ...dz.lambda];
        trace.push(nrm(delta) / nrm(z));
        z = z.map((v, i) => v + delta[i]);
    }
    return {
        z,
        trace,
        converged: trace[trace.length - 1] < 1e-14,
        ldlt: { x: ldlt.x, lambda: ldlt.lambda },
    };
}

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

interface Fixture {
    n: number;
    vertices: Vec3[];
    edges: Edge[];
    op: HierarchicalOperator;
    P: Preconditioner;
    /** `{op, P}` with `σ` scaled -- the A35 item 9 invariance check. */
    withSigmaScale(s: number): { op: HierarchicalOperator; P: Preconditioner };
    /** Dense matrix of the SAME operator (`densify(applyAScalar)`, symmetrised). Lazy. */
    denseOp(): Float64Array;
    /** `assembleAFlat`: the INDEPENDENT dense `Ā`. Lazy. */
    denseA(): Float64Array;
    /** `nFar` of the block source actually used (0 ⇒ the exact limit of A36 item 1). */
    nFar: number;
}
const fixtureCache = new Map<string, Fixture>();
/** `leafSize >= n/4` drives `nFar` to 0 at `DEFAULT_THETA` -- A36 item 1's exact-limit route. */
function fixture(n: number, theta = DEFAULT_THETA, leaf = LEAF): Fixture {
    const key = `${n}/${theta}/${leaf}`;
    const hit = fixtureCache.get(key);
    if (hit) return hit;
    const { vertices, edges } = trefoil(n);
    const bvh = buildBvh(vertices, edges, leaf, EPS);
    const bct = buildBct(bvh, { theta });
    const source = { kind: 'lists', bct } as const;
    const H = prepareA(bvh, n, source, { alpha: ALPHA, beta: BETA, epsilon: EPS });
    const op = makeHierarchicalOperator(H);
    const Lo = prepareLowOrder(vertices, edges, bvh, source, {
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
        nFar: bct.stats.nFar,
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
 * PERTURBED edge-length targets. `edgeLengths(v, e)` returns the CURRENT lengths, so an unperturbed
 * `ℓ⁰` makes `Φ ≡ 0` and the whole table degenerate (A35 item 5). The `sin 3r` term makes `Φ` vary
 * per edge rather than being one global scaling, which a wrong row ORDER could still satisfy.
 */
function perturbedEll0(vertices: Vec3[], edges: Edge[]): number[] {
    return edgeLengths(vertices, edges).map((v, r) => v * (1 + 0.02 * Math.sin(3 * r) + 0.01));
}

type SetName = 'edge' | 'baryedge' | 'edgepin' | 'baryedgepin';
interface SplitSet {
    /** The full stacked `C`, for the DENSE oracle only -- `|E| × 3n` is `3n²` f64 (see DOUBTS). */
    C: number[][];
    phi: number[];
    negPhi: number[];
    rowStart: number;
    otherRows: number[][];
    k: number;
}
function splitSet(name: SetName, fx: Fixture): SplitSet {
    const { vertices, edges } = fx;
    const x0 = barycenterTarget(vertices, edges);
    const x0off: Vec3 = [x0[0] + 0.05, x0[1] - 0.03, x0[2] + 0.02];
    const pre: ConstraintBlock[] =
        name === 'baryedge' || name === 'baryedgepin' ? [barycenterBlock(x0off)] : [];
    const post: ConstraintBlock[] =
        name === 'edgepin' || name === 'baryedgepin' ? [pointBlock(7, [-0.2, 0.1, 0])] : [];
    const set = [...pre, edgeLengthsBlock(perturbedEll0(vertices, edges)), ...post];
    const { C, phi } = evaluateConstraintSet(set, vertices, edges);
    const rowStart = pre.length * 3;
    const otherRows = C.filter((_, r) => r < rowStart || r >= rowStart + edges.length);
    return { C, phi, negPhi: phi.map((v) => -v), rowStart, otherRows, k: C.length };
}

function gradRhs(fx: Fixture): number[] {
    const g = gradientAnalytical(
        fx.vertices,
        fx.edges,
        calculateDisjointPairs(fx.edges),
        ALPHA,
        BETA,
        EPS,
    );
    const b: number[] = [];
    for (let c = 0; c < 3; c++) for (let i = 0; i < fx.n; i++) b.push(g[i][c]);
    return b;
}

// ---------------------------------------------------------------------------
// Dense instruments for the SPD block: an orthonormal basis of `V = null(C_edge)`, and the
// restriction of a densified `3n` operator to it.
// ---------------------------------------------------------------------------

function densify3(dim: number, mv: (u: Float64Array, out: Float64Array) => void): Float64Array {
    const M = new Float64Array(dim * dim);
    const e = new Float64Array(dim);
    const col = new Float64Array(dim);
    for (let j = 0; j < dim; j++) {
        e.fill(0);
        e[j] = 1;
        mv(e, col);
        for (let i = 0; i < dim; i++) M[i * dim + j] = col[i];
    }
    return M;
}
function matmul(X: Float64Array, Y: Float64Array, d: number): Float64Array {
    const Z = new Float64Array(d * d);
    for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) {
            let s = 0;
            for (let q = 0; q < d; q++) s += X[i * d + q] * Y[q * d + j];
            Z[i * d + j] = s;
        }
    }
    return Z;
}
function symDefect(X: Float64Array, d: number): number {
    let mx = 0;
    let mv = 0;
    for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) {
            mv = Math.max(mv, Math.abs(X[i * d + j]));
            mx = Math.max(mx, Math.abs(X[i * d + j] - X[j * d + i]));
        }
    }
    return mv > 0 ? mx / mv : 0;
}
/** Modified Gram-Schmidt on `P`'s columns, one re-orthogonalisation pass. */
function nullBasis(Pd: Float64Array, dim: number, want: number): Float64Array[] {
    const Z: Float64Array[] = [];
    for (let j = 0; j < dim && Z.length < want; j++) {
        const v = new Float64Array(dim);
        for (let i = 0; i < dim; i++) v[i] = Pd[i * dim + j];
        for (let pass = 0; pass < 2; pass++) {
            for (const q of Z) {
                let d = 0;
                for (let i = 0; i < dim; i++) d += q[i] * v[i];
                for (let i = 0; i < dim; i++) v[i] -= d * q[i];
            }
        }
        const nv = nrm(v);
        if (nv > 1e-8) {
            for (let i = 0; i < dim; i++) v[i] /= nv;
            Z.push(v);
        }
    }
    return Z;
}
function restrict(X: Float64Array, Z: Float64Array[], dim: number): Float64Array {
    const d = Z.length;
    const R = new Float64Array(d * d);
    const XZ = Z.map((z) => {
        const o = new Float64Array(dim);
        for (let i = 0; i < dim; i++) {
            let s = 0;
            for (let q = 0; q < dim; q++) s += X[i * dim + q] * z[q];
            o[i] = s;
        }
        return o;
    });
    for (let a = 0; a < d; a++) {
        for (let b = 0; b < d; b++) {
            let s = 0;
            for (let i = 0; i < dim; i++) s += Z[a][i] * XZ[b][i];
            R[a * d + b] = s;
        }
    }
    return R;
}

/**
 * The route's `project`, re-derived here from an INDEPENDENT tangent formula and an INDEPENDENT
 * dense `C_edge` (the shipped `edgeLengthsBlock` rows), so the SPD block does not measure
 * `nullspace.ts` against itself (memory `common-mode-blindness`). `O(n³)`; n ≤ 48 only.
 */
function densePd(fx: Fixture, ell0: number[]): { Pd: Float64Array; Ce: number[][] } {
    const { n, edges } = fx;
    const dim = 3 * n;
    const Ce = evaluateConstraintSet([edgeLengthsBlock(ell0)], fx.vertices, edges).C;
    const m = Ce.length;
    const CCt = new Float64Array(m * m);
    for (let r = 0; r < m; r++) {
        for (let s = 0; s < m; s++) {
            let v = 0;
            for (let j = 0; j < dim; j++) v += Ce[r][j] * Ce[s][j];
            CCt[r * m + s] = v;
        }
    }
    const LL = cholesky(Float64Array.from(CCt), m);
    const solveCC = (rhs: Float64Array): Float64Array => {
        const x = Float64Array.from(rhs);
        for (let i = 0; i < m; i++) {
            let s = x[i];
            for (let q = 0; q < i; q++) s -= LL[i * m + q] * x[q];
            x[i] = s / LL[i * m + i];
        }
        for (let i = m - 1; i >= 0; i--) {
            let s = x[i];
            for (let q = i + 1; q < m; q++) s -= LL[q * m + i] * x[q];
            x[i] = s / LL[i * m + i];
        }
        return x;
    };
    const Pd = new Float64Array(dim * dim);
    const col = new Float64Array(m);
    for (let j = 0; j < dim; j++) {
        for (let r = 0; r < m; r++) col[r] = Ce[r][j];
        const y = solveCC(col);
        for (let i = 0; i < dim; i++) {
            let s = 0;
            for (let r = 0; r < m; r++) s += Ce[r][i] * y[r];
            Pd[i * dim + j] = (i === j ? 1 : 0) - s;
        }
    }
    return { Pd, Ce };
}

// ===========================================================================
// 1. §8.3 ITEM 2 -- SPD ON THE CONSTRAINED SUBSPACE. NOTHING BELOW QUOTES A
//    CONVERGENCE NUMBER UNTIL THIS BLOCK HAS RUN.
// ===========================================================================

describe('§8.3 item 2: the projected operator is SPD on the constrained subspace', () => {
    const fx = fixture(48);
    const dim = 3 * fx.n;
    const ell0 = perturbedEll0(fx.vertices, fx.edges);
    const { Pd, Ce } = densePd(fx, ell0);
    const Z = nullBasis(Pd, dim, dim - fx.edges.length);

    test('C_edge N = 0 EXACTLY, which is what puts the constant modes inside V', () => {
        let worst = 0;
        for (let r = 0; r < Ce.length; r++) {
            for (let c = 0; c < 3; c++) {
                let s = 0;
                for (let i = 0; i < fx.n; i++) s += Ce[r][c * fx.n + i];
                worst = Math.max(worst, Math.abs(s));
            }
        }
        log('max|C_edge N|', worst);
        // Not a tolerance: `+T_r[c]` and `−T_r[c]` are the only two terms in the block sum.
        expect(worst).toBe(0);
        // And the row is NOT trivially zero, so the cell above is not vacuous.
        let maxEntry = 0;
        for (const row of Ce) for (const v of row) maxEntry = Math.max(maxEntry, Math.abs(v));
        expect(maxEntry).toBeGreaterThan(0.1);
    });

    test('the orthonormal basis of V is complete and really lies in V', () => {
        expect(Z.length).toBe(2 * fx.n);
        let orth = 0;
        for (let a = 0; a < Z.length; a++) {
            for (let b = 0; b < Z.length; b++) {
                let d = 0;
                for (let i = 0; i < dim; i++) d += Z[a][i] * Z[b][i];
                orth = Math.max(orth, Math.abs(d - (a === b ? 1 : 0)));
            }
        }
        expect(orth).toBeLessThan(1e-12);
    });

    test('Zᵀ(P A_σ P)Z is symmetric and POSITIVE DEFINITE; Zᵀ(P Ā P)Z is SINGULAR', () => {
        const Asig = densify3(dim, (u, out) => applyAsigma(fx.op, u, out));
        const Abar = densify3(dim, (u, out) => fx.op.applyA(u, out));
        const RA = restrict(matmul(Pd, matmul(Asig, Pd, dim), dim), Z, dim);
        const RAbar = restrict(matmul(Pd, matmul(Abar, Pd, dim), dim), Z, dim);
        const spA = spectrum(Float64Array.from(RA), Z.length);
        const spAbar = spectrum(Float64Array.from(RAbar), Z.length);
        log('symDefect(Zᵀ P A_σ P Z)', symDefect(RA, Z.length));
        log('λ_min(Zᵀ P A_σ P Z)', spA.lambda1);
        log('cond(Zᵀ P A_σ P Z)', spA.lambdaMax / spA.lambda1);
        log('nullDim(Zᵀ P Ā P Z)', spAbar.nullDim);
        expect(symDefect(RA, Z.length)).toBeLessThan(1e-13);
        // THE SPD assertion. The value that trips it is λ_min <= 0, and it EXISTS one line away:
        expect(spA.lambda1).toBeGreaterThan(0.1);
        // ...the UNSHIFTED projected operator, whose nullity on V is exactly 3 (the constant modes).
        // So the σ-shift is load-bearing, not decoration.
        expect(spAbar.nullDim).toBe(3);
        expect(spAbar.lambda1).toBeLessThan(1e-10);
    });

    test('the preconditioned product has a REAL POSITIVE spectrum on V (the similarity transform)', () => {
        const Asig = densify3(dim, (u, out) => applyAsigma(fx.op, u, out));
        const Minv = densify3(dim, (u, out) => fx.P.apply(u, out));
        const RA = restrict(matmul(Pd, matmul(Asig, Pd, dim), dim), Z, dim);
        const RM = restrict(matmul(Pd, matmul(Minv, Pd, dim), dim), Z, dim);
        const d = Z.length;
        // denseSpectrum.ts `cholesky` TSDoc: with A = L Lᵀ, Lᵀ(M⁻¹A)L⁻ᵀ = Lᵀ M⁻¹ L is SYMMETRIC and
        // similar to the product, so `spectrum` applies to a quantity that is otherwise unreachable.
        const Lf = cholesky(Float64Array.from(RA), d);
        const LtM = new Float64Array(d * d);
        for (let i = 0; i < d; i++) {
            for (let j = 0; j < d; j++) {
                let s = 0;
                for (let q = i; q < d; q++) s += Lf[q * d + i] * RM[q * d + j];
                LtM[i * d + j] = s;
            }
        }
        const Sim = new Float64Array(d * d);
        for (let i = 0; i < d; i++) {
            for (let j = 0; j < d; j++) {
                let s = 0;
                for (let q = j; q < d; q++) s += LtM[i * d + q] * Lf[q * d + j];
                Sim[i * d + j] = s;
            }
        }
        const sp = spectrum(Float64Array.from(Sim), d);
        log('λ_min(preconditioned product on V)', sp.lambda1);
        log('λ_max(preconditioned product on V)', sp.lambdaMax);
        log('symDefect(similarity transform)', symDefect(Sim, d));
        expect(symDefect(Sim, d)).toBeLessThan(1e-12);
        expect(sp.nullDim).toBe(0);
        // λ_min is the constant mode `precond.ts`'s +11ᵀ/(nσ) correction pins at exactly 1, and the
        // constant modes are inside V (first cell of this block), so it shows up here.
        expect(sp.lambda1).toBeGreaterThan(0.99);
        expect(sp.lambda1).toBeLessThan(1.01);
        expect(sp.lambdaMax).toBeGreaterThan(1);
    });

    test('the Euclidean symmetric defect of the PRECONDITIONED product is ~0.1 and MUST BE', () => {
        // A gate written "symDefect(Ã⁻¹A) ≈ 0 on the constrained subspace" rejects the correct
        // order: a product of two SPD matrices is not symmetric. Pinned so nobody "fixes" it.
        const Asig = densify3(dim, (u, out) => applyAsigma(fx.op, u, out));
        const Minv = densify3(dim, (u, out) => fx.P.apply(u, out));
        const PAP = matmul(Pd, matmul(Asig, Pd, dim), dim);
        const PMP = matmul(Pd, matmul(Minv, Pd, dim), dim);
        const defect = symDefect(restrict(matmul(PMP, PAP, dim), Z, dim), Z.length);
        log('symDefect(Zᵀ P Ã⁻¹ P A_σ P Z)', defect);
        expect(defect).toBeGreaterThan(1e-2);
    });
});

// ===========================================================================
// 2. The projector, and §8.2's "do not write a new solver".
// ===========================================================================

describe('the O(n) projector', () => {
    test('C_edge C_edgeᵀ is EXACTLY cyclic tridiagonal, and the diagonal is 2 only to ~4 ulp', () => {
        const fx = fixture(48);
        const { Pd: _unused, Ce } = densePd(fx, perturbedEll0(fx.vertices, fx.edges));
        const m = Ce.length;
        const dim = 3 * fx.n;
        let offBandNonzero = 0;
        let maxOffBand = 0;
        let diagUlp = 0;
        for (let r = 0; r < m; r++) {
            for (let s = 0; s < m; s++) {
                let v = 0;
                for (let j = 0; j < dim; j++) v += Ce[r][j] * Ce[s][j];
                const cyc = Math.min(Math.abs(r - s), m - Math.abs(r - s));
                if (cyc > 1) {
                    if (v !== 0) offBandNonzero++;
                    maxOffBand = Math.max(maxOffBand, Math.abs(v));
                }
                if (r === s) diagUlp = Math.max(diagUlp, Math.abs(v - 2) / Number.EPSILON);
            }
        }
        log('off-band entries with |M| > 0', offBandNonzero);
        log('max|diag − 2| in ulps', diagUlp);
        // Identically zero, not "below 1e-15": the dot product of two non-adjacent rows has no terms.
        expect(offBandNonzero).toBe(0);
        expect(maxOffBand).toBe(0);
        // The brief's "diagonal exactly 2.000000" is a toFixed(6) artefact: it is 2|T_r|².
        expect(diagUlp).toBeGreaterThan(0);
        expect(diagUlp).toBeLessThan(16);
    });

    test('P is idempotent, self-adjoint and annihilates C_edge; the band diagonal is ≈2', () => {
        for (const n of [48, 240]) {
            const fx = fixture(n);
            const sol = prepareNullspace(fx.op, fx.P, {
                vertices: fx.vertices,
                edges: fx.edges,
                rowStart: 0,
            });
            const p = sol.report.projector;
            log(`n=${n} projector`, JSON.stringify(p));
            expect(p.idempotenceDefect).toBeLessThan(1e-13);
            expect(p.annihilationDefect).toBeLessThan(1e-13);
            expect(p.selfAdjointDefect).toBeLessThan(1e-13);
            expect(p.triResidual).toBeLessThan(1e-12);
            expect(Math.abs(p.minDiagonal - 2)).toBeLessThan(1e-12);
            expect(sol.report.kEdge).toBe(n);
            expect(sol.report.kOther).toBe(0);
        }
    });

    test('the SPARSE C_edge matches the shipped dense edgeLengthsBlock rows', () => {
        // The route re-derives the tangents, a SECOND spelling of `edgeLengthsBlock`'s formula. This
        // is the differential that catches a sign, a layout or a row-order error in that spelling:
        // the route's `x` is asserted to satisfy the SHIPPED DENSE rows, which it computed from its
        // own sparse ones. Non-vacuous by the second half: a route given a DIFFERENT (non-uniformly
        // perturbed) geometry at the same `n` builds a different `C_edge`, passes every shape and
        // cycle check (issue utof/repulsive-test2#83 -- "same vertices" has no enforcement anywhere),
        // and then misses the dense rows by O(1).
        const fx = fixture(48);
        const dim = 3 * fx.n;
        const ell0 = perturbedEll0(fx.vertices, fx.edges);
        const Ce = evaluateConstraintSet([edgeLengthsBlock(ell0)], fx.vertices, fx.edges).C;
        const d = lcgVec(fx.edges.length, 7);
        const worstOf = (verts: Vec3[]): number => {
            const res = prepareNullspace(fx.op, fx.P, {
                vertices: verts,
                edges: fx.edges,
                rowStart: 0,
            }).solve(zeros(dim), d);
            let worst = 0;
            for (let r = 0; r < Ce.length; r++) {
                let acc = 0;
                for (let j = 0; j < dim; j++) acc += Ce[r][j] * res.x[j];
                worst = Math.max(worst, Math.abs(acc - d[r]));
            }
            return worst;
        };
        const clean = worstOf(fx.vertices);
        const skewed: Vec3[] = fx.vertices.map((v, i) => [
            v[0] + 0.3 * Math.sin(5 * i),
            v[1],
            v[2] - 0.2 * Math.cos(3 * i),
        ]);
        const wrong = worstOf(skewed);
        log('max|C_edge_dense x − d| clean', clean);
        log('max|C_edge_dense x − d| wrong geometry, same n', wrong);
        expect(clean).toBeLessThan(1e-12);
        expect(wrong).toBeGreaterThan(1e-2);
    });
});

// ===========================================================================
// 3. Accuracy -- the bar, as amended (A35 item 1), and the gauge (A35 item 3 / A36 item 3).
// ===========================================================================

describe('accuracy against the DD-refined truth (A35 item 1)', () => {
    for (const n of [48, 240]) {
        for (const name of ['baryedge', 'baryedgepin', 'edgepin'] as const) {
            test(`n=${n} ${name}: x and λ within ${BAR} of truth, on grad/−φ and gen/−φ`, () => {
                const fx = fixture(n);
                const dim = 3 * n;
                const S = splitSet(name, fx);
                const A = fx.denseOp();
                const sol = prepareNullspace(
                    fx.op,
                    fx.P,
                    { vertices: fx.vertices, edges: fx.edges, rowStart: S.rowStart },
                    S.otherRows,
                );
                expect(sol.report.schur.k).toBe(S.otherRows.length);
                expect(sol.report.kEdge).toBe(n);
                // A35 item 5: the targets are offset, so the bottom rhs is non-trivial.
                expect(Math.max(...S.phi.map(Math.abs))).toBeGreaterThan(1e-3);
                for (const [rhsName, top] of [
                    ['grad/−φ', gradRhs(fx)],
                    ['gen/−φ', lcgVec(dim, 99)],
                ] as const) {
                    const truth = refinedTruth(A, n, S.C, top, S.negPhi);
                    expect(truth.converged).toBe(true);
                    const xT = truth.z.slice(0, dim);
                    const lT = truth.z.slice(dim);
                    const mine = sol.solve(top, S.negPhi);
                    const ex = relErr(mine.x, xT);
                    const el = dist(mine.lambda, lT) / Math.max(1, nrm(lT));
                    const ldlt = solveSaddleFromA(A, n, S.C, top, S.negPhi, 'ldlt');
                    log(
                        `n=${n} ${name} ${rhsName}`,
                        `x=${ex.toExponential(2)} λ=${el.toExponential(2)} ` +
                            `ldlt=${relErr(ldlt.x, xT).toExponential(2)} ` +
                            `iters=${mine.gmres.iters} η=${mine.gmres.eta.toExponential(2)} ` +
                            `residual=${mine.residual.toExponential(2)} ` +
                            `edgeRes=${mine.edgeConstraintResidual.toExponential(2)}`,
                    );
                    expect(ex).toBeLessThan(BAR);
                    expect(el).toBeLessThan(BAR);
                    // The O(n) rows are satisfied, which is the whole point of the projection.
                    expect(mine.edgeConstraintResidual).toBeLessThan(1e-11);
                    expect(mine.lambda.length).toBe(S.k);
                    expect(Array.isArray(mine.lambda)).toBe(true);
                    expect(Array.isArray(mine.x)).toBe(true);
                }
            });
        }
    }

    test('the pure edgeLengths set is TRANSLATION-SINGULAR: no truth, no dense reference for x', () => {
        const n = 240;
        const fx = fixture(n);
        const S = splitSet('edge', fx);
        const A = fx.denseOp();
        const top = gradRhs(fx);
        // (a) the DD refinement DIVERGES -- so A35's amended bar has no instrument here.
        const truth = refinedTruth(A, n, S.C, top, S.negPhi);
        log('pure-edge DD refine trace', truth.trace.map((v) => v.toExponential(1)).join(' '));
        expect(truth.converged).toBe(false);
        expect(truth.trace[3]).toBeGreaterThan(0.1);
        // (b) the two dense factorisations disagree about x, and agree mod translation.
        const ldlt = solveSaddleFromA(A, n, S.C, top, S.negPhi, 'ldlt');
        const lu = solveSaddleFromA(A, n, S.C, top, S.negPhi, 'lu');
        log('pure-edge LU vs LDLᵀ literal', relErr(lu.x, ldlt.x));
        log('pure-edge LU vs LDLᵀ modTrans', relErr(demean(lu.x, n), demean(ldlt.x, n)));
        expect(relErr(lu.x, ldlt.x)).toBeGreaterThan(0.1);
        expect(relErr(demean(lu.x, n), demean(ldlt.x, n))).toBeLessThan(1e-9);
        // (c) the route agrees with both MOD TRANSLATION, and its own means are ~0 (the A_σ gauge
        //     with a mean-free gradient rhs). The LITERAL comparison is asserted to FAIL, so the
        //     demeaning is proven load-bearing rather than assumed (A36 item 3's reading rule).
        const sol = prepareNullspace(fx.op, fx.P, {
            vertices: fx.vertices,
            edges: fx.edges,
            rowStart: 0,
        });
        const mine = sol.solve(top, S.negPhi);
        log('pure-edge route literal vs LDLᵀ', relErr(mine.x, ldlt.x));
        log('pure-edge route modTrans vs LDLᵀ', relErr(demean(mine.x, n), demean(ldlt.x, n)));
        expect(relErr(mine.x, ldlt.x)).toBeGreaterThan(0.1);
        expect(relErr(demean(mine.x, n), demean(ldlt.x, n))).toBeLessThan(BAR);
        expect(Math.max(...blockSums(mine.x, n).map(Math.abs))).toBeLessThan(1e-12);
        expect(Math.max(...blockSums(ldlt.x, n).map(Math.abs))).toBeGreaterThan(1);
    });

    test('the gauge is exactly Nᵀx = Nᵀb/σ, and a mean-carrying rhs is INCONSISTENT not merely gauged', () => {
        const n = 48;
        const fx = fixture(n);
        const dim = 3 * n;
        const S = splitSet('edge', fx);
        const sol = prepareNullspace(fx.op, fx.P, {
            vertices: fx.vertices,
            edges: fx.edges,
            rowStart: 0,
        });
        const gen = lcgVec(dim, 99);
        const mine = sol.solve(gen, S.negPhi);
        const want = blockSums(gen, n).map((s) => s / fx.op.sigma);
        const got = blockSums(mine.x, n);
        log('Nᵀx', got.map((v) => v.toExponential(3)).join(', '));
        log('Nᵀb/σ', want.map((v) => v.toExponential(3)).join(', '));
        for (let c = 0; c < 3; c++) {
            expect(Math.abs(got[c] - want[c])).toBeLessThan(1e-9 * Math.abs(want[c]));
        }
        // Non-vacuity: `Nᵀb/σ` is not ~0 on this rhs, so the identity is discriminating.
        expect(Math.max(...want.map(Math.abs))).toBeGreaterThan(1e-3);
        // ...and the saddle residual FLAGS the inconsistency (`[N; 0]` is in the dense kernel, so a
        // rhs with `Nᵀb ≠ 0` has no solution at all). The same rhs mean-free is clean.
        log('pure-edge residual, mean-carrying rhs', mine.residual);
        expect(mine.residual).toBeGreaterThan(1e-3);
        const clean = sol.solve(demean(gen, n), S.negPhi);
        log('pure-edge residual, mean-free rhs', clean.residual);
        expect(clean.residual).toBeLessThan(1e-9);
    });

    test('σ-invariance: exact on a pinned set, exactly 1/σ on the gauge (A35 item 9)', () => {
        const n = 48;
        const fx = fixture(n);
        const dim = 3 * n;
        const top = lcgVec(dim, 99);
        for (const name of ['baryedge', 'edge'] as const) {
            const S = splitSet(name, fx);
            const eb = { vertices: fx.vertices, edges: fx.edges, rowStart: S.rowStart };
            const base = prepareNullspace(fx.op, fx.P, eb, S.otherRows).solve(top, S.negPhi);
            const xs: number[][] = [];
            for (const s of [4, 0.25]) {
                const { op, P } = fx.withSigmaScale(s);
                xs.push(prepareNullspace(op, P, eb, S.otherRows).solve(top, S.negPhi).x);
            }
            const litt = xs.map((x) => relErr(x, base.x));
            const dem = xs.map((x) => relErr(demean(x, n), demean(base.x, n)));
            const ratios = xs.map((x) =>
                blockSums(x, n).map((v, c) => v / blockSums(base.x, n)[c]),
            );
            log(`σ-invariance ${name} literal`, litt.map((v) => v.toExponential(2)).join(' '));
            log(`σ-invariance ${name} demeaned`, dem.map((v) => v.toExponential(2)).join(' '));
            log(`σ-invariance ${name} mean ratios`, JSON.stringify(ratios));
            // The mean-free part is σ-free on EVERY set -- that is the falsifier for the reduction.
            for (const v of dem) expect(v).toBeLessThan(1e-10);
            if (name === 'baryedge') {
                // Pinned: σ cancels entirely, translation included.
                for (const v of litt) expect(v).toBeLessThan(1e-10);
                for (const r of ratios)
                    for (const v of r) expect(Math.abs(v - 1)).toBeLessThan(1e-9);
            } else {
                // Gauge: the per-block means scale as EXACTLY 1/σ. This is the cell that makes the
                // pinned one above non-vacuous -- the same code, a different structural answer.
                for (const v of litt) expect(v).toBeGreaterThan(1e-4);
                expect(Math.abs(ratios[0][0] - 0.25)).toBeLessThan(1e-6);
                expect(Math.abs(ratios[1][0] - 4)).toBeLessThan(1e-5);
            }
        }
    });
});

// ===========================================================================
// 4. A36 item 1: the compressed cell PAIRED with its `nFar = 0` exact limit -- the only cell that
//    can see a plumbing defect, because both other cells are dominated by θ's 1.6e-3…4.5e-3.
// ===========================================================================

describe('A36 item 1: compressed vs nFar = 0 exact limit, against the INDEPENDENT dense Ā', () => {
    test('n=240: the exact limit is 1.3e9 tighter than the compressed cell', () => {
        const n = 240;
        const compressed = fixture(n, DEFAULT_THETA, LEAF);
        // `leafSize >= n/4` leaves 4 leaves, so no block can be admissible: nFar = 0 and `A_bct` IS
        // `Ā` up to summation order. 8b's two independent routes to nFar = 0 (`theta: 0` and a big
        // leaf) agreed to 4 digits; this picks the cheap one (`theta: 0` streams, which is O(N²)).
        const exact = fixture(n, DEFAULT_THETA, Math.ceil(n / 4));
        expect(compressed.nFar).toBeGreaterThan(100);
        expect(exact.nFar).toBe(0);
        const results: Record<string, number> = {};
        for (const [label, fx] of [
            ['compressed', compressed],
            ['exact', exact],
        ] as const) {
            const S = splitSet('baryedge', fx);
            const sol = prepareNullspace(
                fx.op,
                fx.P,
                { vertices: fx.vertices, edges: fx.edges, rowStart: S.rowStart },
                S.otherRows,
            );
            const top = gradRhs(fx);
            const mine = sol.solve(top, S.negPhi);
            // The oracle here is the INDEPENDENT dense assembly, not the operator's own densify.
            const ldlt = solveSaddleFromA(fx.denseA(), n, S.C, top, S.negPhi, 'ldlt');
            results[label] = relErr(mine.x, ldlt.x);
            log(`nFar=${fx.nFar} relErr vs assembleAFlat`, results[label]);
        }
        // The exact-limit bar. Measured 2.3435e-12 -- 1e-10 is carry 3's floor-respecting bar at
        // n=240 and it is met with 43x of margin.
        expect(results.exact).toBeLessThan(BAR);
        // Non-vacuity, and the reason the pair exists: the SAME assertion on the compressed cell is
        // 1.3e9 looser, so a compressed-only suite could not see a plumbing defect of up to ~3e-3.
        expect(results.compressed).toBeGreaterThan(1e-4);
        expect(results.compressed / results.exact).toBeGreaterThan(1e6);
    });

    test('at n=48 the DEFAULT leaf already gives nFar = 0 — so the small cells are exact, not compressed', () => {
        // Recorded so nobody reads the n=48 cells of this file as evidence about θ.
        expect(fixture(48).nFar).toBe(0);
        expect(fixture(240).nFar).toBeGreaterThan(100);
    });
});

// ===========================================================================
// 5. Knock-outs. Every gate above gets its tripping value here.
// ===========================================================================

describe('knock-outs', () => {
    const n = 48;
    const fx = fixture(n);
    const dim = 3 * n;

    test("'precond-unprojected' (the DISQUALIFIED order) converges happily and breaks the constraints", () => {
        // §8.3 item 2's failure mode, end to end: the iterate leaves V. η is FINE, the iteration
        // count is FINE, and only the constraint residual sees it. This is the cell that makes the
        // SPD/order block above non-vacuous on the SHIPPED path rather than only in dense algebra.
        const S = splitSet('baryedge', fx);
        const eb = { vertices: fx.vertices, edges: fx.edges, rowStart: S.rowStart };
        const A = fx.denseOp();
        const top = gradRhs(fx);
        const ldlt = solveSaddleFromA(A, n, S.C, top, S.negPhi, 'ldlt');
        const clean = prepareNullspace(fx.op, fx.P, eb, S.otherRows).solve(top, S.negPhi);
        const ko = prepareNullspace(fx.op, fx.P, eb, S.otherRows, {
            knockOut: true,
            defect: 'precond-unprojected',
            etaAccept: 1,
        }).solve(top, S.negPhi);
        log('KO precond-unprojected edgeRes', ko.edgeConstraintResidual);
        log('KO precond-unprojected η', ko.gmres.eta);
        log('KO precond-unprojected relErr', relErr(ko.x, ldlt.x));
        expect(clean.edgeConstraintResidual).toBeLessThan(1e-13);
        expect(ko.edgeConstraintResidual).toBeGreaterThan(1e-3);
        // THE POINT: the η certificate PASSES on the broken order, by 11 orders.
        expect(ko.gmres.eta).toBeLessThan(1e-10);
        expect(ko.gmres.iters).toBe(clean.gmres.iters);
        expect(relErr(ko.x, ldlt.x)).toBeGreaterThan(1e-2);
    });

    test("'skip-particular' passes η too, and only the constraint residual sees it", () => {
        const S = splitSet('baryedge', fx);
        const eb = { vertices: fx.vertices, edges: fx.edges, rowStart: S.rowStart };
        const ko = prepareNullspace(fx.op, fx.P, eb, S.otherRows, {
            knockOut: true,
            defect: 'skip-particular',
            etaAccept: 1,
        }).solve(gradRhs(fx), S.negPhi);
        log('KO skip-particular edgeRes', ko.edgeConstraintResidual);
        log('KO skip-particular η', ko.gmres.eta);
        expect(ko.edgeConstraintResidual).toBeGreaterThan(1e-3);
        expect(ko.gmres.eta).toBeLessThan(1e-10);
    });

    test("forwarded 'naive-shift' is caught by x and the residual, and is INVISIBLE to edgeRes", () => {
        // The constant-mode reduction survives the projection (STAGE 0a); removing it moves `x` by
        // O(1). It leaves the edge rows exactly satisfied, so the two gates are complementary and
        // neither is redundant.
        const S = splitSet('baryedge', fx);
        const eb = { vertices: fx.vertices, edges: fx.edges, rowStart: S.rowStart };
        const A = fx.denseOp();
        const top = gradRhs(fx);
        const ldlt = solveSaddleFromA(A, n, S.C, top, S.negPhi, 'ldlt');
        const ko = prepareNullspace(fx.op, fx.P, eb, S.otherRows, {
            knockOut: true,
            schurDefect: 'naive-shift',
        }).solve(top, S.negPhi);
        log('KO naive-shift relErr', relErr(ko.x, ldlt.x));
        log('KO naive-shift residual', ko.residual);
        log('KO naive-shift edgeRes', ko.edgeConstraintResidual);
        expect(relErr(ko.x, ldlt.x)).toBeGreaterThan(0.1);
        expect(ko.residual).toBeGreaterThan(0.1);
        expect(ko.edgeConstraintResidual).toBeLessThan(1e-13);
    });

    test("'lambda-sign' is BLIND on grad/−φ at kOther = 3 and CAUGHT on gen/−φ (A35 item 5)", () => {
        // A35 item 5, reproduced on this route and sharpened. At kOther = 3 the solvability condition
        // `Gᵀλ = Nᵀb` has a SQUARE invertible `G`, so a mean-free top forces `λ_O(1) = 0` and a
        // flipped multiplier sign is invisible. The `gen/−φ` cell is therefore REQUIRED. At
        // kOther = 6 (bary + pin) `G` is 6×3, λ is no longer determined by that condition, and the
        // blindness disappears -- which A35 did not state.
        const table: string[] = [];
        for (const name of ['baryedge', 'baryedgepin'] as const) {
            const S = splitSet(name, fx);
            const eb = { vertices: fx.vertices, edges: fx.edges, rowStart: S.rowStart };
            for (const [rhsName, top] of [
                ['grad/−φ', gradRhs(fx)],
                ['gen/−φ', lcgVec(dim, 99)],
            ] as const) {
                const clean = prepareNullspace(fx.op, fx.P, eb, S.otherRows).solve(top, S.negPhi);
                const ko = prepareNullspace(fx.op, fx.P, eb, S.otherRows, {
                    knockOut: true,
                    schurDefect: 'lambda-sign',
                }).solve(top, S.negPhi);
                const sep = relErr(ko.x, clean.x);
                const lamO = clean.lambda.filter((_, r) => r < S.rowStart || r >= S.rowStart + n);
                table.push(
                    `${name}/${rhsName}: |λ_O(1)|=${nrm(lamO).toExponential(2)} sep=${sep.toExponential(2)}`,
                );
                if (name === 'baryedge' && rhsName === 'grad/−φ') {
                    expect(nrm(lamO)).toBeLessThan(1e-12);
                    expect(sep).toBeLessThan(1e-13); // BLIND
                } else {
                    expect(nrm(lamO)).toBeGreaterThan(1e-3);
                    expect(sep).toBeGreaterThan(1e-3); // CAUGHT
                }
                // The edge multipliers are never trivial, on any cell.
                expect(nrm(clean.lambda.slice(S.rowStart, S.rowStart + n))).toBeGreaterThan(1);
            }
        }
        log('lambda-sign visibility', table.join(' | '));
    });
});

// ===========================================================================
// 6. Cost -- §8.1 is the whole reason this route exists, so the counts are structural integers.
// ===========================================================================

describe('cost (§8.1): the |E| rows cost ZERO extra GMRES runs', () => {
    test('prepare runs exactly kOther GMRES runs; the pure edgeLengths set runs none', () => {
        const fx = fixture(48);
        for (const name of ['edge', 'baryedge', 'baryedgepin'] as const) {
            const S = splitSet(name, fx);
            const sol = prepareNullspace(
                fx.op,
                fx.P,
                {
                    vertices: fx.vertices,
                    edges: fx.edges,
                    rowStart: S.rowStart,
                },
                S.otherRows,
            );
            log(
                `${name} prepare`,
                `kEdge=${sol.report.kEdge} kOther=${sol.report.kOther} ` +
                    `GMRES runs=${sol.report.schur.columns.length} ` +
                    `iters=[${sol.report.schur.columns.map((c) => c.iters).join(',')}]`,
            );
            expect(sol.report.schur.columns.length).toBe(S.otherRows.length);
            expect(sol.report.kEdge).toBe(48);
            // `k = |E|` would be 48 GMRES runs here -- the O(N²·iters) cost §8.1 measures.
            expect(sol.report.schur.columns.length).toBeLessThanOrEqual(SCHUR_MAX_K);
        }
    });

    test('one solve = ONE GMRES, and the projection count is an exact structural integer', () => {
        const fx = fixture(48);
        const S = splitSet('baryedge', fx);
        const sol = prepareNullspace(
            fx.op,
            fx.P,
            {
                vertices: fx.vertices,
                edges: fx.edges,
                rowStart: S.rowStart,
            },
            S.otherRows,
        );
        for (const [label, top] of [
            ['grad/−φ', gradRhs(fx)],
            ['gen/−φ', lcgVec(3 * 48, 99)],
        ] as const) {
            const before = sol.report.projections;
            const r = sol.solve(top, S.negPhi);
            const delta = sol.report.projections - before;
            // 2 per operator apply × (the iteration's matvecs + the η check + prepareSchur's own
            // discarded saddle residual) + 2 per preconditioner apply + 3 outside the iteration
            // (`x_p`, `b̂`, `λ_edge`). Left preconditioning ⇒ precondApplies = iters + 1, so this is
            // also exactly `4·iters + 9`.
            const expected = 2 * (r.gmres.matvecs + 2) + 2 * r.gmres.precondApplies + 3;
            log(
                `${label} projections`,
                `delta=${delta} expected=${expected} iters=${r.gmres.iters} ` +
                    `matvecs=${r.gmres.matvecs} precondApplies=${r.gmres.precondApplies}`,
            );
            expect(delta).toBe(expected);
            expect(delta).toBe(4 * r.gmres.iters + 9);
            // ONE GMRES: the recorded run is the only one, and it is not the degenerate zero-rhs
            // short-circuit (a projection iterate has `rhsTop = 0` but `b̂ = −P Ā x_p ≠ 0`).
            expect(r.gmres.iters).toBeGreaterThan(1);
            expect(r.gmres.stop).not.toBe('zero-rhs');
        }
        // A projection iterate: rhsTop = 0, rhsBottom = −Φ. The Schur route spends ZERO applies on
        // this case; the nullspace route spends ONE GMRES, because `x_p` has to be mapped through Ā.
        const before = sol.report.projections;
        const proj = sol.solve(zeros(3 * 48), S.negPhi);
        log(
            'projection iterate (rhsTop = 0)',
            `iters=${proj.gmres.iters} projections=${sol.report.projections - before}`,
        );
        expect(proj.gmres.iters).toBeGreaterThan(1);
        expect(proj.gmres.stop).not.toBe('zero-rhs');
    });

    test('the projected operator costs no extra GMRES iterations', () => {
        // Recorded, not barred: the preconditioned κ_eig ratio (projected / unprojected) saturates
        // at 1.12 (measured densely at n ≤ 240), and the iteration counts follow.
        for (const n of [48, 240]) {
            const fx = fixture(n);
            const top = gradRhs(fx);
            const projected = prepareNullspace(fx.op, fx.P, {
                vertices: fx.vertices,
                edges: fx.edges,
                rowStart: 0,
            }).solve(top, zeros(n));
            const plain = gmres(fx.op, fx.P, top, { tol: DEFAULT_SCHUR_TOL });
            log(
                `n=${n} iteration counts`,
                `projected=${projected.gmres.iters} unprojected=${plain.iters} ` +
                    `ratio=${(projected.gmres.iters / plain.iters).toFixed(3)}`,
            );
            expect(projected.gmres.iters).toBeLessThan(2 * plain.iters);
        }
    });
});

// ===========================================================================
// 7. The O(n) residual, and the refusals.
// ===========================================================================

describe('the structural O(n) saddle residual', () => {
    test('reproduces operatorSaddleResidual, and a wrong λ sign moves it to O(1)', () => {
        // `operatorSaddleResidual` needs the |E| × 3n dense `C` this route exists not to build, so
        // the route recomputes the same quantity structurally. Same formula, different summation
        // order, so the agreement is relative and bounded by cancellation (the residual is a
        // difference of O(1) terms landing at 1e-12).
        const fx = fixture(48);
        for (const name of ['edge', 'baryedge'] as const) {
            const S = splitSet(name, fx);
            const sol = prepareNullspace(
                fx.op,
                fx.P,
                {
                    vertices: fx.vertices,
                    edges: fx.edges,
                    rowStart: S.rowStart,
                },
                S.otherRows,
            );
            const top = gradRhs(fx);
            const mine = sol.solve(top, S.negPhi);
            const rhs = [...top, ...S.negPhi];
            const ref = operatorSaddleResidual(fx.op, S.C, [...mine.x, ...mine.lambda], rhs);
            const tripper = operatorSaddleResidual(
                fx.op,
                S.C,
                [...mine.x, ...mine.lambda.map((v) => -v)],
                rhs,
            );
            log(
                `${name} residual agreement`,
                `mine=${mine.residual.toExponential(3)} ref=${ref.toExponential(3)} ` +
                    `rel=${(Math.abs(mine.residual - ref) / ref).toExponential(2)} ` +
                    `negated-λ ref=${tripper.toExponential(3)}`,
            );
            expect(Math.abs(mine.residual - ref) / ref).toBeLessThan(1e-3);
            // Non-vacuity: the same comparison against a negated λ is O(1) relative.
            expect(Math.abs(mine.residual - tripper) / tripper).toBeGreaterThan(0.1);
        }
    });
});

describe('refusals (all typed, all trippable)', () => {
    const fx = fixture(48);
    const dim = 3 * fx.n;
    const eb = () => ({ vertices: fx.vertices, edges: fx.edges, rowStart: 0 });

    test('the clean configuration is accepted, so the refusals below are not vacuous', () => {
        expect(() => prepareNullspace(fx.op, fx.P, eb())).not.toThrow();
    });

    test('an edge list that is not ONE cycle in edge order is refused, naming the dense path', () => {
        const cases: [string, Edge[], string][] = [
            ['open chain', fx.edges.slice(0, -1), 'DENSE path'],
            [
                'permuted edge order',
                (() => {
                    const e = fx.edges.map((x) => [...x] as Edge);
                    [e[1], e[5]] = [e[5], e[1]];
                    return e;
                })(),
                'DENSE path',
            ],
            [
                // A malformed index, not a routing decision: there is no dense path to recommend
                // for an edge list that does not describe a graph, so this message names the edge.
                'self-loop',
                (() => {
                    const e = fx.edges.map((x) => [...x] as Edge);
                    e[2] = [2, 2];
                    return e;
                })(),
                'self-loop',
            ],
        ];
        for (const [label, edges, expected] of cases) {
            let caught: unknown;
            try {
                prepareNullspace(fx.op, fx.P, { vertices: fx.vertices, edges, rowStart: 0 });
            } catch (e) {
                caught = e;
            }
            log(`refusal ${label}`, (caught as Error)?.message?.slice(0, 120));
            expect(caught).toBeInstanceOf(NullspaceRouteError);
            expect((caught as NullspaceRouteError).reason).toBe('cycle');
            // The message must name the decision the caller has to make (route it densely), or,
            // for a malformed graph, the structure that is wrong.
            expect((caught as Error).message).toContain(expected);
            // And it must be a SolveNotCertifiedError, or `optimizer.ts` folds it into
            // 'singular_system' -- a routing mistake reported as a singular matrix (A33, A35 item 6).
            expect(caught).toBeInstanceOf(SolveNotCertifiedError);
        }
    });

    test('shape mismatches are refused', () => {
        const bad: [string, () => unknown][] = [
            [
                'vertex count',
                () =>
                    prepareNullspace(fx.op, fx.P, {
                        vertices: fx.vertices.slice(0, -1),
                        edges: fx.edges,
                        rowStart: 0,
                    }),
            ],
            ['short otherRow', () => prepareNullspace(fx.op, fx.P, eb(), [zeros(dim - 1)])],
        ];
        // `rowStart = 0` with `kOther = 0` is legal (the pure edge set); the illegal one is
        // `rowStart > kOther`, which would leave a gap in the stacked row block.
        expect(() =>
            prepareNullspace(fx.op, fx.P, {
                vertices: fx.vertices,
                edges: fx.edges,
                rowStart: 1,
            }),
        ).toThrow(NullspaceRouteError);
        for (const [label, fn] of bad) {
            let caught: unknown;
            try {
                fn();
            } catch (e) {
                caught = e;
            }
            log(`refusal ${label}`, (caught as Error)?.message?.slice(0, 100));
            expect(caught).toBeInstanceOf(NullspaceRouteError);
            expect((caught as NullspaceRouteError).reason).toBe('shape');
        }
    });

    test('too many O(1) rows is refused by maxK, which is a real knob', () => {
        const rows = Array.from({ length: 4 }, (_, r) => {
            const row = zeros(dim);
            row[r] = 1;
            return row;
        });
        expect(() => prepareNullspace(fx.op, fx.P, eb(), rows, { maxK: 3 })).toThrow(
            NullspaceRouteError,
        );
        expect(() => prepareNullspace(fx.op, fx.P, eb(), rows, { maxK: 4 })).not.toThrow();
    });

    test('a defect without knockOut is refused', () => {
        expect(() =>
            prepareNullspace(fx.op, fx.P, eb(), [], { defect: 'skip-particular' }),
        ).toThrow(/knock-out instrument/);
        expect(() =>
            prepareNullspace(fx.op, fx.P, eb(), [], { schurDefect: 'naive-shift' }),
        ).toThrow(/knock-out instrument/);
    });

    test('a degenerate (zero-length) edge zeroes its C_edge row and is reported, not inverted', () => {
        // `edgeLengthsBlock`'s own TSDoc: a degenerate edge zeroes its row → singular saddle → the
        // 'singular_system' rejection path is the backstop. Here the zero row makes
        // `C_edge C_edgeᵀ` singular and `solveLaplacian`'s zero-pivot throw reports it. A plain
        // Error is CORRECT for this one: it really is a singular system.
        const verts = fx.vertices.map((v) => [...v] as Vec3);
        verts[3] = [...verts[2]] as Vec3;
        expect(() =>
            prepareNullspace(fx.op, fx.P, { vertices: verts, edges: fx.edges, rowStart: 0 }),
        ).toThrow(/zero pivot/);
    });

    test('totalLength + edgeLengths is refused AT SET CONSTRUCTION, so this route never sees it', () => {
        // Carry 6, verified rather than assumed: the total-length row is exactly the SUM of the edge
        // rows, so stacking them is an EXACT rank deficiency. This route would NOT notice --
        // `C_edge C_edgeᵀ` is formed from the edge rows ALONE and stays SPD -- so the construction
        // guard is what protects it. `dispatch.ts` `dispatchDescentStep` calls this guard.
        expect(() =>
            assertValidConstraintSet([
                totalLengthBlock(1),
                edgeLengthsBlock(perturbedEll0(fx.vertices, fx.edges)),
            ]),
        ).toThrow(/mutually exclusive/);
        // Non-vacuous: each alone is accepted.
        expect(() => assertValidConstraintSet([totalLengthBlock(1)])).not.toThrow();
        expect(() =>
            assertValidConstraintSet([edgeLengthsBlock(perturbedEll0(fx.vertices, fx.edges))]),
        ).not.toThrow();
    });
});

// ===========================================================================
// 8. n = 960: the amended bar where it matters, and the oracle's own failure (A35 item 1).
// ===========================================================================

describe('n = 960 (the amended bar, and the oracle over it)', () => {
    test('bary+edge, grad/−φ: within 1e-10 of truth while dense LDLᵀ is NOT', () => {
        const n = 960;
        const fx = fixture(n);
        const dim = 3 * n;
        const S = splitSet('baryedge', fx);
        const A = fx.denseOp();
        const top = gradRhs(fx);
        const truth = refinedTruth(A, n, S.C, top, S.negPhi);
        expect(truth.converged).toBe(true);
        const xT = truth.z.slice(0, dim);
        const lT = truth.z.slice(dim);
        const sol = prepareNullspace(
            fx.op,
            fx.P,
            {
                vertices: fx.vertices,
                edges: fx.edges,
                rowStart: S.rowStart,
            },
            S.otherRows,
        );
        const mine = sol.solve(top, S.negPhi);
        // The dense LDLᵀ answer is the refinement's own starting point: `refinedTruth` ran exactly
        // `solveSaddleFromA(A, n, S.C, top, S.negPhi, 'ldlt')`, so re-running it was a bit-identical
        // ~27 s repeat. The dense-LU solve that used to follow fed one log line and no assertion
        // (~45 s), and is gone. (test-suite review 2026-11-24, README.md "Tests")
        const ldlt = truth.ldlt;
        const mineErr = relErr(mine.x, xT);
        const ldltErr = relErr(ldlt.x, xT);
        log('n=960 route vs truth', mineErr);
        log('n=960 dense LDLᵀ vs truth', ldltErr);
        log('n=960 route literal vs LDLᵀ', relErr(mine.x, ldlt.x));
        log('n=960 λ vs truth (relative)', dist(mine.lambda, lT) / Math.max(1, nrm(lT)));
        log(
            'n=960 iters / η / edgeRes',
            `${mine.gmres.iters} / ${mine.gmres.eta} / ${mine.edgeConstraintResidual}`,
        );
        // THE BAR, as amended by A35 item 1.
        expect(mineErr).toBeLessThan(BAR);
        // ...and the reason the amendment exists: the oracle row 9 originally named is itself over
        // the bar here, so a literal "reproduce the dense path to 1e-10 at n=960" cell would have
        // measured the oracle. MEASURED 1.32e-10 against the route's 3.79e-12.
        expect(ldltErr).toBeGreaterThan(BAR);
        expect(mineErr).toBeLessThan(ldltErr / 10);
        // λ is LOGGED, not barred, at n=960: the edge multipliers are recovered by a post-solve
        // `(C_edge C_edgeᵀ)⁻¹ C_edge (b − Ā x − C_oᵀ λ_o)` whose error carries
        // `cond(C_edge C_edgeᵀ) ~ n²` on top of `x`'s -- MEASURED 4.24e-10 at n=960 against
        // 4.06e-12 at n=240, a 68x gap over a 4x in n. The n ≤ 240 cells bar it.
        expect(mine.edgeConstraintResidual).toBeLessThan(1e-10);
    });
});
