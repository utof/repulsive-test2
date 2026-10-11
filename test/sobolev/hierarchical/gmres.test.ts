/**
 * Spec §11.2 task 7 of Phase 2H, first half: `gmres.ts` — left-preconditioned GMRES, its `η`
 * probe, and the INVALID clauses of §7.3 — everything that does NOT depend on the
 * preconditioner's mass weight. The H2/H3 gate calibration over the n-ladder (the bar's
 * non-vacuity, `τ_η = 1e-8`, `G(n)`, median-of-5, the A9 settlement) is NOT here: it is measured
 * against the operator that ships, after the mass weight is settled (spec amendment A14,
 * @issue utof/repulsive-test2#72). So NO iteration count and NO `η` floor of a real operator is an
 * expectation in this file — they appear only in `console.log` lines as data.
 *
 * TWO FIXTURE FAMILIES, and why both:
 *  - SYNTHETIC (`synth`): a weighted complete-graph Laplacian plus the rank-1 shift, `n = 8`,
 *    `dim = 24`, f64, with a dense `A_σ` that is the ORACLE. Here GMRES has exact-arithmetic
 *    theory to be held to (termination in `<= n` steps, the recurrence residual EQUAL to the true
 *    one under `M = I`, the Givens bookkeeping against an independent least-squares solve), and a
 *    hard numeric expectation is safe. The preconditioner is injected, so a perturbed-weights
 *    inverse stands in for `Ã⁻¹` — a genuinely non-trivial SPD `M` whose mass weight is nobody's
 *    business.
 *  - REAL (`real`): the hierarchical `A_bct` and the shipped `Ã⁻¹` on the repo's `trefoil(n)`
 *    (UNnormalised, bbox extent 5.4716). Used for what only the real objects can show: the
 *    structural application counters against `Preconditioner.solves`, `A_σ 1 = σ 1`, the shift
 *    against a densified `A_bct`, and the direction of the recurrence-vs-true gap.
 *
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.3, §5.6, §7.3, §7.4, §11.2 task 7
 * @issue utof/repulsive-test2#72
 */
import { describe, expect, setDefaultTimeout, test } from 'bun:test';
import { trefoil } from '../../../src/core/fixtures';
import { applyA, applyAScalar, prepareA } from '../../../src/core/sobolev/hierarchical/apply';
import { buildBct } from '../../../src/core/sobolev/hierarchical/bct';
import { buildBvh } from '../../../src/core/sobolev/hierarchical/bvh';
import {
    applyAsigma,
    backwardError,
    DEFAULT_PROBE_SCHEDULE,
    defaultMaxit,
    type EtaClause,
    type EtaProbeOptions,
    type EtaProbeRecord,
    type EtaProbeReport,
    estimateNorm2,
    etaViolations,
    type GmresOptions,
    type GmresResult,
    type GmresStop,
    gmres,
    probeIterations,
    U_F32,
    U_F64,
} from '../../../src/core/sobolev/hierarchical/gmres';
import { prepareLowOrder } from '../../../src/core/sobolev/hierarchical/lowOrder';
import {
    type HierarchicalOperator,
    makeHierarchicalOperator,
} from '../../../src/core/sobolev/hierarchical/operator';
import { makePreconditioner } from '../../../src/core/sobolev/hierarchical/precond';
import { cholesky, cholSolve, densify, spectrum } from './denseSpectrum';

setDefaultTimeout(600_000);

const ALPHA = 3;
const BETA = 6;
const EPS_A = 1e-10;
const LEAF = 8;
/** Production θ (spec §4.1). The χ=0 exact operator is built separately, below. */
const THETA = 0.5;

const log = (name: string, v: unknown) => console.log(`[gmres] ${name} = ${v}`);

/** `scratch/curves.ts` `lcg` — deterministic, no `Math.random`. */
function lcg(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 4294967296;
    };
}
const randVec = (len: number, seed: number): Float64Array => {
    const rnd = lcg(seed);
    const v = new Float64Array(len);
    for (let i = 0; i < len; i++) v[i] = rnd() - 0.5;
    return v;
};
const norm = (x: ArrayLike<number>): number => {
    let s = 0;
    for (let i = 0; i < x.length; i++) s += x[i] * x[i];
    return Math.sqrt(s);
};
const dotp = (x: ArrayLike<number>, y: ArrayLike<number>): number => {
    let s = 0;
    for (let i = 0; i < x.length; i++) s += x[i] * y[i];
    return s;
};
/** `‖a − b‖`. */
const distance = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
    return Math.sqrt(s);
};

// ---------------------------------------------------------------------------
// Synthetic fixture: the oracle is a dense matrix.
// ---------------------------------------------------------------------------

interface Synth {
    n: number;
    /** Scalar `A_σ = A0 + (σ/n) 11ᵀ`, row-major — the ORACLE. */
    As: Float64Array;
    sigma: number;
    /** `HierarchicalOperator` over the 3n coordinate-block layout, `applyA = blockdiag(A0)`. */
    op: HierarchicalOperator;
    /** The same `A0` applied column-by-column: equal in exact arithmetic, rounded differently. */
    exactApplyA: (u: ArrayLike<number>, out: Float64Array) => void;
}

/**
 * A weighted complete-graph Laplacian: SPD after the shift, distinct eigenvalues generically.
 * `spread` makes the edge weights log-uniform on `[lo, hi]` instead of uniform on `[0.1, 10]`: a
 * wider spectrum, so the Krylov space is NOT exhausted in a handful of steps (the orthogonality
 * of the basis is only a gate while hsub is not tiny).
 */
function synth(n: number, seed: number, spread?: { lo: number; hi: number }): Synth {
    const rnd = lcg(seed);
    const A0 = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            const w = spread ? spread.lo * (spread.hi / spread.lo) ** rnd() : 0.1 + 9.9 * rnd();
            A0[i * n + i] += w;
            A0[j * n + j] += w;
            A0[i * n + j] -= w;
            A0[j * n + i] -= w;
        }
    }
    let tr = 0;
    for (let i = 0; i < n; i++) tr += A0[i * n + i];
    const sigma = tr / n;
    const As = new Float64Array(A0);
    for (let i = 0; i < n * n; i++) As[i] += sigma / n;
    const op: HierarchicalOperator = {
        n,
        sigma,
        applyA: (u, out) => {
            for (let c = 0; c < 3; c++) {
                for (let i = 0; i < n; i++) {
                    let acc = 0;
                    for (let j = 0; j < n; j++) acc += A0[i * n + j] * u[c * n + j];
                    out[c * n + i] = acc;
                }
            }
        },
    };
    const exactApplyA = (u: ArrayLike<number>, out: Float64Array): void => {
        out.fill(0);
        for (let c = 0; c < 3; c++) {
            for (let j = 0; j < n; j++) {
                const uj = u[c * n + j];
                for (let i = 0; i < n; i++) out[c * n + i] += A0[i * n + j] * uj;
            }
        }
    };
    return { n, As, sigma, op, exactApplyA };
}

/** `Ã⁻¹ = (As')⁻¹` for a DIFFERENT synthetic `As'`: an SPD, non-trivial, injected preconditioner. */
function inversePrecond(S: Synth, perturbedSeed: number, spread?: { lo: number; hi: number }) {
    const P = synth(S.n, perturbedSeed, spread);
    const L = cholesky(P.As, S.n);
    const col = new Float64Array(S.n);
    return {
        n: S.n,
        apply: (R: ArrayLike<number>, out: Float64Array): void => {
            for (let c = 0; c < 3; c++) {
                for (let i = 0; i < S.n; i++) col[i] = R[c * S.n + i];
                cholSolve(L, S.n, col);
                for (let i = 0; i < S.n; i++) out[c * S.n + i] = col[i];
            }
        },
    };
}
const identityPrecond = (n: number) => ({
    n,
    apply: (R: ArrayLike<number>, out: Float64Array): void => {
        for (let i = 0; i < 3 * n; i++) out[i] = R[i];
    },
});
/** `y = As x` blockwise, for the oracle residuals. */
function denseMatvec(S: Synth, x: ArrayLike<number>): Float64Array {
    const out = new Float64Array(3 * S.n);
    for (let c = 0; c < 3; c++) {
        for (let i = 0; i < S.n; i++) {
            let acc = 0;
            for (let j = 0; j < S.n; j++) acc += S.As[i * S.n + j] * x[c * S.n + j];
            out[c * S.n + i] = acc;
        }
    }
    return out;
}
/** `x* = A_σ⁻¹ b` by dense Cholesky, blockwise. */
function denseSolve(S: Synth, b: ArrayLike<number>): Float64Array {
    const L = cholesky(S.As, S.n);
    const out = new Float64Array(3 * S.n);
    const col = new Float64Array(S.n);
    for (let c = 0; c < 3; c++) {
        for (let i = 0; i < S.n; i++) col[i] = b[c * S.n + i];
        cholSolve(L, S.n, col);
        for (let i = 0; i < S.n; i++) out[c * S.n + i] = col[i];
    }
    return out;
}

/**
 * An 'exact' operator that differs from the iteration's by a relative 2^-30, the way the χ=0
 * matvec differs from the θ matvec. Clause 3 is about a residual formed with THE SAME operator;
 * two roundings of one operator (`Synth.exactApplyA`) land on the f64 floor, where bit-equal η is
 * a coin flip rather than a bug (MEASURED: it fired in the control).
 */
function perturbedExact(S: Synth): (u: ArrayLike<number>, out: Float64Array) => void {
    return (u, out) => {
        S.exactApplyA(u, out);
        for (let i = 0; i < out.length; i++) out[i] *= 1 + 2 ** -30;
    };
}

// ---------------------------------------------------------------------------
// Real fixture: the hierarchical operator, the shipped preconditioner, and the χ=0 reference.
// ---------------------------------------------------------------------------

const realCache = new Map<number, ReturnType<typeof buildReal>>();
function buildReal(n: number) {
    const { vertices, edges } = trefoil(n);
    const bvh = buildBvh(vertices, edges, LEAF, EPS_A);
    const source = { kind: 'lists', bct: buildBct(bvh, { theta: THETA }) } as const;
    const H = prepareA(bvh, n, source, { alpha: ALPHA, beta: BETA, epsilon: EPS_A });
    const op = makeHierarchicalOperator(H);
    const Lo = prepareLowOrder(vertices, edges, bvh, source, {
        alpha: ALPHA,
        beta: BETA,
        variant: 'vertex',
    });
    const P = makePreconditioner(vertices, edges, Lo, op.sigma);
    // The χ=0 exact reference: a STREAMING source at θ = 0 (spec §7.1). Only its `applyA` is
    // used; its own `sigma` is irrelevant because gmres.ts forms the shift from `op.sigma` for
    // BOTH matvecs.
    const Hx = prepareA(
        bvh,
        n,
        { kind: 'stream', params: { theta: 0 } },
        { alpha: ALPHA, beta: BETA, epsilon: EPS_A },
    );
    const exactApplyA = (u: ArrayLike<number>, out: Float64Array): void => applyA(Hx, u, out);
    return { n, H, op, P, exactApplyA };
}
function real(n: number) {
    let v = realCache.get(n);
    if (!v) {
        v = buildReal(n);
        realCache.set(n, v);
    }
    return v;
}

// ===========================================================================
// 1. The instruments, with no solver in the loop.
// ===========================================================================

/** The stopping iterate's record: the probe always covers it (gmres.ts, terminal probe). */
function lastRecord(res: GmresResult): EtaProbeRecord | undefined {
    const recs = res.probe?.records ?? [];
    return recs[recs.length - 1];
}
/** Every stop reason the loop can report, asserted to be reachable below. */
const STOPS: GmresStop[] = ['tol', 'eta-target', 'breakdown', 'maxit', 'zero-rhs'];

describe('probe schedule (spec §7.3)', () => {
    test('the default is head 80, stride 20, and 600 iterations probe exactly 106 of them', () => {
        expect(DEFAULT_PROBE_SCHEDULE).toEqual({ head: 80, stride: 20 });
        const p = probeIterations(600);
        // §7.3: "80 + 26 = 106 probes" at N = 30 000 — the cost arithmetic of the ruling itself.
        expect(p.length).toBe(106);
        expect(p.slice(0, 80)).toEqual(Array.from({ length: 80 }, (_, i) => i + 1));
        expect(p.slice(80)).toEqual(Array.from({ length: 26 }, (_, i) => 100 + 20 * i));
    });

    test('the EXACT index list, not its length, for a small schedule that has both stages', () => {
        expect(probeIterations(25, { head: 5, stride: 4 })).toEqual([
            1, 2, 3, 4, 5, 9, 13, 17, 21, 25,
        ]);
        // maxit inside the head: the tail is empty and the head is clipped.
        expect(probeIterations(3, { head: 5, stride: 4 })).toEqual([1, 2, 3]);
        // The tail is anchored at `head`, so a head that is not a multiple of the stride is honest.
        expect(probeIterations(20, { head: 6, stride: 5 })).toEqual([1, 2, 3, 4, 5, 6, 11, 16]);
    });

    test('rejects a schedule that could not be recorded as data', () => {
        expect(() => probeIterations(10, { head: -1, stride: 4 })).toThrow();
        expect(() => probeIterations(10, { head: 5, stride: 0 })).toThrow();
        expect(() => probeIterations(10, { head: 5.5, stride: 4 })).toThrow();
    });
});

describe('maxit budget (spec §7.3)', () => {
    test('maxit = min(600, 2·dim), the budget E1 ran with', () => {
        expect(defaultMaxit(3 * 8)).toBe(48);
        expect(defaultMaxit(3 * 64)).toBe(384);
        expect(defaultMaxit(3 * 200)).toBe(600);
        expect(defaultMaxit(3 * 960)).toBe(600);
    });
});

describe('unit roundoff constants', () => {
    test('u_f32 = 2^-24 (the bar 50·u_f32 is 2.98e-6) and u_f64 = 2^-53', () => {
        expect(U_F32).toBe(2 ** -24);
        expect(U_F64).toBe(2 ** -53);
        expect(Number((50 * U_F32).toPrecision(3))).toBe(2.98e-6);
    });
});

describe('η formula (spec §7.3, Rigal-Gaches 1967)', () => {
    test('η = ‖r‖ / (‖A‖‖x‖ + ‖b‖) — BOTH denominator terms are present', () => {
        // Chosen so each term is distinguishable: dropping ‖b‖ gives 0.5, dropping ‖A‖‖x‖ gives 0.25.
        expect(backwardError(1, 3, 1, 1)).toBe(0.25);
        expect(backwardError(2, 3, 1, 1)).toBe(0.5);
        expect(backwardError(1, 1, 3, 1)).toBe(0.25);
        // ‖A‖ and ‖x‖ enter as a PRODUCT, not a sum.
        expect(backwardError(1, 2, 3, 4)).toBe(1 / (2 * 3 + 4));
    });
});

describe('INVALID clauses of H2 (spec §7.3), each fired on its own', () => {
    const base = { etaExact: 1e-7, etaBct: 2e-7, lowerBound: 5, normA2: 10 };
    const v = (p: Partial<typeof base>, uOp: number): EtaClause[] =>
        etaViolations({ ...base, ...p }, uOp);

    test('a healthy probe violates nothing', () => {
        expect(v({}, U_F32)).toEqual([]);
    });

    test('clause 1: η == 0 exactly, and ONLY clause 1', () => {
        expect(v({ etaExact: 0 }, U_F32)).toEqual(['eta-zero']);
        expect(v({ etaExact: Number.MIN_VALUE }, U_F32)).toEqual(['eta-below-uop']);
    });

    test('clause 2: η < u_op/10 — the /10, the direction, and the SCOPE to the operator precision', () => {
        const edge = U_F32 / 10; // 5.96e-9
        expect(v({ etaExact: edge * 0.99 }, U_F32)).toEqual(['eta-below-uop']);
        expect(v({ etaExact: edge * 1.01 }, U_F32)).toEqual([]);
        // The f64 case #59's unscoped `η < u_f32/10` rules INVALID: a correct f64 run reaches
        // etaFloorMin 2.047e-16..6.001e-16 (spec §7.3, MEASURED). 3e-16 < u_f32/10 but not < u_f64/10.
        expect(v({ etaExact: 3e-16 }, U_F64)).toEqual([]);
        expect(v({ etaExact: 3e-16 }, U_F32)).toEqual(['eta-below-uop']);
        // ... and the f64 clause still bites below ITS roundoff.
        expect(v({ etaExact: (U_F64 / 10) * 0.5 }, U_F64)).toEqual(['eta-below-uop']);
    });

    test('clause 3: η_exact bit-equal to η_bct, and NOT merely close', () => {
        expect(v({ etaBct: base.etaExact }, U_F32)).toEqual(['exact-equals-bct']);
        const nextUp = base.etaExact * (1 + 2 ** -52);
        expect(nextUp).not.toBe(base.etaExact);
        expect(v({ etaBct: nextUp }, U_F32)).toEqual([]);
    });

    test('clause 4: the power estimate below the free exact lower bound', () => {
        expect(v({ normA2: 4.999 }, U_F32)).toEqual(['norm-below-lower-bound']);
        expect(v({ normA2: 5 }, U_F32)).toEqual([]); // equal is NOT below
        expect(v({ normA2: 5.001 }, U_F32)).toEqual([]);
    });

    test('a non-finite η is its own violation, not a silent pass', () => {
        // NaN fails every `<` and `===`, so without this clause all four above read "valid".
        expect(v({ etaExact: Number.NaN }, U_F32)).toEqual(['non-finite']);
        expect(v({ etaBct: Number.POSITIVE_INFINITY }, U_F32)).toEqual(['non-finite']);
    });
});

// ===========================================================================
// 2. The shift A_σ = A + (σ/n) 11ᵀ  (spec §2.3).
// ===========================================================================

describe('applyAsigma — the rank-1 shift', () => {
    test('A_σ 1 = σ 1 BIT-EXACTLY on the real operator (A_bct 1 = 0 exactly, apply.ts)', () => {
        // n = 48, chosen because `(σ/n)·n !== σ` there (MEASURED: it is `===` at 10 of the 12 n
        // tried in 40..100, among them 60 and 90, and trivially at every power of two such as 64):
        // at those n the test could not tell `σ·(Σu/n)` from `(σ/n)·Σu`. The guard below makes
        // that a checked property of the fixture, so a fixture drift fails loudly instead of
        // silently removing the test's power.
        const n = 48;
        const { op } = real(n);
        expect((op.sigma / n) * n).not.toBe(op.sigma);
        const ones = new Float64Array(3 * n).fill(1);
        const out = new Float64Array(3 * n);
        applyAsigma(op, ones, out);
        for (let i = 0; i < out.length; i++) expect(out[i]).toBe(op.sigma);
    });

    test('equals a densified A_bct plus σ/n on every entry, per BLOCK (no cross-block coupling)', () => {
        const n = 48;
        const { H, op } = real(n);
        const Ab = densify(n, (u, out) => applyAScalar(H, u, out));
        const As = new Float64Array(Ab);
        for (let i = 0; i < n * n; i++) As[i] += op.sigma / n;
        const dim = 3 * n;
        const e = new Float64Array(dim);
        const col = new Float64Array(dim);
        let worst = 0;
        let leak = 0;
        for (let c = 0; c < 3; c++) {
            for (let j = 0; j < n; j++) {
                e.fill(0);
                e[c * n + j] = 1;
                applyAsigma(op, e, col);
                for (let blk = 0; blk < 3; blk++) {
                    for (let i = 0; i < n; i++) {
                        const got = col[blk * n + i];
                        if (blk === c) worst = Math.max(worst, Math.abs(got - As[i * n + j]));
                        else leak = Math.max(leak, Math.abs(got));
                    }
                }
            }
        }
        log('shift vs dense, max abs entry error', worst);
        log('shift cross-block leak', leak);
        expect(leak).toBe(0);
        // MEASURED 8.9e-16 (an ulp-scale rounding of `Ab_ij + σ/n`); bound 4.5x that.
        expect(worst).toBeLessThan(4e-15);
    });
});

describe('estimateNorm2 — the 30-iteration power estimate (spec §7.3)', () => {
    test('30 matvecs, the history never decreases, and it never exceeds λ_max', () => {
        const S = synth(8, 3);
        const mv = (u: Float64Array, out: Float64Array) => applyAsigma(S.op, u, out);
        const est = estimateNorm2(mv, 24);
        expect(est.matvecs).toBe(30);
        expect(est.history.length).toBe(30);
        for (let k = 1; k < est.history.length; k++) {
            expect(est.history[k]).toBeGreaterThanOrEqual(est.history[k - 1]);
        }
        const lamMax = spectrum(S.As, 8).lambdaMax;
        log('power estimate / λ_max', est.norm / lamMax);
        expect(est.norm).toBeLessThanOrEqual(lamMax * (1 + 1e-14));
        expect(est.norm).toBe(est.history[29]);
        // MEASURED 0.99987 on this fixture; the bound is 4x the shortfall, not 0.9.
        expect(est.norm).toBeGreaterThan(0.9995 * lamMax);
    });
});

// ===========================================================================
// 3. The algorithm in exact-arithmetic regimes (synthetic, f64).
// ===========================================================================

function solveSynth(
    S: Synth,
    P: { n: number; apply: (R: ArrayLike<number>, out: Float64Array) => void },
    b: Float64Array,
    opts: Partial<GmresOptions> = {},
) {
    return gmres(S.op, P, b, { tol: 1e-13, ...opts });
}

describe('exact-arithmetic convergence (synthetic, M = I)', () => {
    test('terminates in <= n steps (n distinct eigenvalues), true residual at the f64 floor', () => {
        const S = synth(8, 7);
        const b = randVec(24, 11);
        const res = solveSynth(S, identityPrecond(8), b);
        log('iters (M = I, n = 8)', res.iters);
        expect(res.iters).toBeLessThanOrEqual(8);
        const r = distance(b, denseMatvec(S, res.x)) / norm(b);
        log('true rel residual', r);
        expect(r).toBeLessThan(5e-15); // MEASURED 4.8e-16
        const star = denseSolve(S, b);
        const errStar = distance(res.x, star) / norm(star);
        log('M=I x error vs dense solve', errStar);
        expect(errStar).toBeLessThan(5e-15); // MEASURED 5.2e-16
    });

    test('the recurrence residual is monotone non-increasing EXACTLY, and equals the true residual when M = I', () => {
        const S = synth(8, 7);
        const b = randVec(24, 11);
        const res = solveSynth(S, identityPrecond(8), b, {
            tol: 0,
            maxit: 6,
            probe: { exactApplyA: S.exactApplyA, uOp: U_F64, schedule: { head: 6, stride: 1 } },
        });
        for (let k = 1; k < res.recurrence.length; k++) {
            expect(res.recurrence[k]).toBeLessThanOrEqual(res.recurrence[k - 1]);
        }
        // M = I ⇒ the recurrence residual IS the true relative residual, iterate by iterate.
        const recs = res.probe?.records ?? [];
        expect(recs.length).toBe(6);
        let worst = 0;
        for (const rec of recs) {
            const truth = rec.resNormExact / norm(b);
            worst = Math.max(worst, Math.abs(truth - res.recurrence[rec.iter - 1]) / truth);
        }
        log('M=I: worst |true - recurrence| / true over 6 iterates', worst);
        expect(worst).toBeLessThan(1e-11); // MEASURED 1.3e-12
        // The true residual is monotone too (GMRES minimises it over a growing space).
        for (let k = 1; k < recs.length; k++) {
            expect(recs[k].resNormExact).toBeLessThanOrEqual(
                recs[k - 1].resNormExact * (1 + 1e-12),
            );
        }
    });

    test('an EXACT inverse as the left preconditioner converges in exactly one iteration', () => {
        const S = synth(8, 7);
        const exact = inversePrecond(S, 7); // same seed ⇒ same A_σ ⇒ Ã⁻¹ = A_σ⁻¹
        const b = randVec(24, 11);
        const res = solveSynth(S, exact, b);
        expect(res.iters).toBe(1);
        expect(res.matvecs).toBe(1);
        expect(res.precondApplies).toBe(2); // Ã⁻¹b once, then once per iteration
        const star = denseSolve(S, b);
        const errStar = distance(res.x, star) / norm(star);
        log('exact-inverse x error vs dense solve', errStar);
        expect(errStar).toBeLessThan(2e-15); // MEASURED 2.3e-16
    });
});

describe('scale invariance (GMRES is invariant to a scalar scaling of Ã⁻¹)', () => {
    test('Ã⁻¹ x 1e-20 and x 1e+20 give the same iteration count, recurrence and x̃', () => {
        const S = synth(8, 7);
        const P = inversePrecond(S, 21);
        const scaled = (c: number) => ({
            n: P.n,
            apply: (R: ArrayLike<number>, out: Float64Array) => {
                P.apply(R, out);
                for (let i = 0; i < out.length; i++) out[i] *= c;
            },
        });
        const b = randVec(24, 11);
        // tol 1e-6 stops at iteration 7 (recurrence 1.9e-7), above the ~1e-16 noise floor where the
        // last entry of the recurrence is rounding, not signal (MEASURED: 7e-2 relative there).
        const ref = solveSynth(S, P, b, { tol: 1e-6 });
        expect(ref.iters).toBe(7);
        for (const c of [1e-20, 1e20]) {
            const r = solveSynth(S, scaled(c), b, { tol: 1e-6 });
            expect(r.iters).toBe(ref.iters);
            let worst = 0;
            for (let k = 0; k < ref.iters; k++) {
                worst = Math.max(
                    worst,
                    Math.abs(r.recurrence[k] - ref.recurrence[k]) / ref.recurrence[k],
                );
            }
            log(`scale ${c}: worst recurrence relative difference`, worst);
            expect(worst).toBeLessThan(1e-11); // MEASURED 2.3e-13
            expect(distance(r.x, ref.x) / norm(ref.x)).toBeLessThan(1e-11);
        }
    });
});

describe('Arnoldi / Givens structure (synthetic, non-trivial left preconditioner)', () => {
    const WIDE = { lo: 0.01, hi: 100 };
    const S = synth(12, 7, WIDE);
    const P = inversePrecond(S, 21, WIDE);
    const b = randVec(36, 5);
    const K = 6;
    const res = gmres(S.op, P, b, { tol: 0, maxit: K, retainArnoldi: true });
    const dim = 36;
    const V = res.arnoldi?.basis ?? [];
    const H = res.arnoldi?.hessenberg ?? [];

    test('the basis is orthonormal to roundoff', () => {
        expect(V.length).toBe(K + 1);
        let worst = 0;
        for (let i = 0; i < V.length; i++) {
            for (let j = 0; j <= i; j++) {
                worst = Math.max(worst, Math.abs(dotp(V[i], V[j]) - (i === j ? 1 : 0)));
            }
        }
        log('max |VᵀV − I|', worst);
        // MEASURED 1.26e-14 at K = 6 on a spread spectrum (it is 7e-16 at K = 2: MGS loses ~u·κ
        // per step). A dropped `v_k` in the orthogonalisation is O(1e-1), 10 decades above.
        expect(worst).toBeLessThan(6e-14);
    });

    test('M V_k = V_{k+1} H_k with M = Ã⁻¹ A_σ, and H is upper Hessenberg', () => {
        expect(H.length).toBe(K);
        let worst = 0;
        const t = new Float64Array(dim);
        const w = new Float64Array(dim);
        for (let j = 0; j < K; j++) {
            expect(H[j].length).toBe(j + 2); // column j has entries 0..j+1 only: Hessenberg
            applyAsigma(S.op, V[j], t);
            P.apply(t, w);
            const rebuilt = new Float64Array(dim);
            for (let i = 0; i <= j + 1; i++) {
                for (let q = 0; q < dim; q++) rebuilt[q] += H[j][i] * V[i][q];
            }
            worst = Math.max(worst, distance(w, rebuilt) / norm(w));
        }
        log('max relative defect of M V = V H', worst);
        expect(worst).toBeLessThan(1e-15); // MEASURED 1.7e-16
    });

    test('the Givens recurrence residual equals an independent least-squares solve of min‖β e1 − H y‖', () => {
        const beta = res.beta;
        let worst = 0;
        for (let k = 1; k <= K; k++) {
            // Normal equations on the (k+1)×k Hessenberg: small and well-conditioned here.
            const G = new Float64Array(k * k);
            const rhs = new Float64Array(k);
            for (let i = 0; i < k; i++) {
                for (let j = 0; j < k; j++) {
                    let s = 0;
                    for (let q = 0; q <= Math.min(i, j) + 1; q++)
                        s += (H[i][q] ?? 0) * (H[j][q] ?? 0);
                    G[i * k + j] = s;
                }
                rhs[i] = beta * H[i][0];
            }
            const L = cholesky(G, k);
            cholSolve(L, k, rhs);
            const resid = new Float64Array(k + 1);
            resid[0] = beta;
            for (let j = 0; j < k; j++) {
                for (let q = 0; q <= j + 1; q++) resid[q] -= H[j][q] * rhs[j];
            }
            const ls = norm(resid) / beta;
            worst = Math.max(worst, Math.abs(ls - res.recurrence[k - 1]) / ls);
        }
        log('worst |LS − recurrence| / LS', worst);
        expect(worst).toBeLessThan(1e-14); // MEASURED 1.4e-15
    });
});

// ===========================================================================
// 4. Stop reasons, censoring, and input guards.
// ===========================================================================

describe('stop reasons and censoring (spec §5.3, §7.4: compare against maxit, NEVER against n)', () => {
    test('tol stop: iters < maxit, not censored, and tol / maxit are RECORDED on the result', () => {
        const S = synth(8, 7);
        const b = randVec(24, 11);
        const res = solveSynth(S, inversePrecond(S, 21), b, { tol: 1e-6 });
        expect(res.stop).toBe('tol');
        expect(res.tol).toBe(1e-6);
        expect(res.maxit).toBe(defaultMaxit(24));
        expect(res.recurrence.length).toBe(res.iters);
        expect(res.b).toEqual(b);
        expect(res.b).not.toBe(b); // a copy, so the record cannot drift if the caller reuses `b`
        expect(res.recurrence[res.iters - 1]).toBeLessThanOrEqual(1e-6);
        expect(res.iters).toBeLessThan(res.maxit);
        expect(res.censored).toBe(false);
        // The rule is about the TOLERANCE having been reached, not the stored vector's identity.
        if (res.iters > 1) expect(res.recurrence[res.iters - 2]).toBeGreaterThan(1e-6);
    });

    test('the tol comparison is INCLUSIVE: tol equal to the iteration-3 recurrence stops at 3', () => {
        const S = synth(8, 7);
        const P = inversePrecond(S, 21);
        const b = randVec(24, 11);
        const ref = solveSynth(S, P, b, { tol: 0, maxit: 6 });
        const edge = ref.recurrence[2];
        const res = solveSynth(S, P, b, { tol: edge });
        expect(res.iters).toBe(3);
        expect(res.stop).toBe('tol');
    });

    test('maxit exhausted: censored even though iters < n', () => {
        const S = synth(8, 7);
        const b = randVec(24, 11);
        const res = solveSynth(S, identityPrecond(8), b, { tol: 0, maxit: 3 });
        expect(res.iters).toBe(3);
        expect(res.iters).toBeLessThan(S.n);
        expect(res.stop).toBe('maxit');
        expect(res.censored).toBe(true);
    });

    test('NOT censored although iters >= n (n = 2 converges in 2 steps with maxit = 12)', () => {
        const S = synth(2, 9);
        const b = randVec(6, 4);
        const res = solveSynth(S, identityPrecond(2), b, { tol: 1e-12, maxit: 12 });
        expect(res.iters).toBeGreaterThanOrEqual(S.n);
        expect(res.iters).toBeLessThan(12);
        expect(res.censored).toBe(false);
    });

    test('happy breakdown: an exact inverse exhausts the Krylov space at iteration 1', () => {
        const S = synth(8, 7);
        const res = solveSynth(S, inversePrecond(S, 7), randVec(24, 11), { tol: 0 });
        expect(res.iters).toBe(1);
        expect(res.stop).toBe('breakdown');
        expect(res.censored).toBe(false);
    });

    test('every stop reason is reachable (a reason no test can produce is a dead branch)', () => {
        const S = synth(8, 7);
        const P = inversePrecond(S, 21);
        const b = randVec(24, 11);
        const seen = new Set<GmresStop>([
            solveSynth(S, P, b, { tol: 1e-6 }).stop,
            gmres(S.op, P, b, {
                tol: 0,
                probe: {
                    exactApplyA: S.exactApplyA,
                    uOp: U_F64,
                    schedule: { head: 20, stride: 1 },
                    etaTarget: 1e-3,
                },
            }).stop,
            solveSynth(S, inversePrecond(S, 7), b, { tol: 0 }).stop,
            solveSynth(S, identityPrecond(8), b, { tol: 0, maxit: 3 }).stop,
            solveSynth(S, identityPrecond(8), new Float64Array(24)).stop,
        ]);
        expect([...seen].sort()).toEqual([...STOPS].sort());
    });

    test('b = 0: the solution is 0, no iteration runs, nothing is applied beyond the rhs', () => {
        const S = synth(8, 7);
        const res = solveSynth(S, identityPrecond(8), new Float64Array(24));
        expect(res.iters).toBe(0);
        expect(res.stop).toBe('zero-rhs');
        expect(res.x.every((v) => v === 0)).toBe(true);
        expect(res.matvecs).toBe(0);
        expect(res.censored).toBe(false);
    });

    test('a preconditioner that annihilates b is an error, not a NaN carried through the loop', () => {
        const S = synth(8, 7);
        const zero = { n: 8, apply: (_R: ArrayLike<number>, out: Float64Array) => out.fill(0) };
        expect(() => gmres(S.op, zero, randVec(24, 1), { tol: 1e-6 })).toThrow(/annihilated/);
    });

    test('input guards', () => {
        const S = synth(8, 7);
        const P = identityPrecond(8);
        const b = randVec(24, 1);
        expect(() => gmres(S.op, P, randVec(23, 1), { tol: 1e-6 })).toThrow(/length/);
        expect(() => gmres(S.op, identityPrecond(9), b, { tol: 1e-6 })).toThrow(/n=/);
        expect(() => gmres(S.op, P, b, { tol: -1 })).toThrow(/tol/);
        expect(() => gmres(S.op, P, b, { tol: Number.NaN })).toThrow(/tol/);
        expect(() => gmres(S.op, P, b, { tol: 1e-6, maxit: 0 })).toThrow(/maxit/);
        expect(() => gmres(S.op, P, b, { tol: 1e-6, maxit: 2.5 })).toThrow(/maxit/);
        expect(() =>
            gmres(S.op, P, b, { tol: 1e-6, probe: { exactApplyA: S.op.applyA, uOp: 0 } }),
        ).toThrow(/uOp/);
    });
});

// ===========================================================================
// 5. Left vs right: right preconditioning is a KNOCK-OUT instrument (spec §5.3, §11.2 row 7).
// ===========================================================================

describe('side selection', () => {
    const S = synth(8, 7);
    const P = inversePrecond(S, 21);
    const b = randVec(24, 5);
    const probe = { exactApplyA: S.exactApplyA, uOp: U_F64, schedule: { head: 6, stride: 1 } };

    test("side: 'right' is REFUSED without knockOut: true (the house guard, precond.ts / apply.ts)", () => {
        expect(() => gmres(S.op, P, b, { tol: 1e-6, side: 'right' })).toThrow(/knock-out/);
        expect(() => gmres(S.op, P, b, { tol: 1e-6, side: 'left', knockOut: false })).not.toThrow();
        expect(() => gmres(S.op, P, b, { tol: 1e-6, side: 'right', knockOut: true })).not.toThrow();
    });

    test('the default side is left, and the result records the side it ran', () => {
        expect(gmres(S.op, P, b, { tol: 1e-6 }).side).toBe('left');
        expect(gmres(S.op, P, b, { tol: 1e-6, side: 'right', knockOut: true }).side).toBe('right');
    });

    test('RIGHT: the recurrence residual IS the true residual; LEFT: it is not (spec §5.3)', () => {
        const worstGap = (side: 'left' | 'right'): number => {
            const res = gmres(S.op, P, b, {
                tol: 0,
                maxit: 6,
                side,
                knockOut: side === 'right',
                probe,
            });
            let worst = 0;
            for (const rec of res.probe?.records ?? []) {
                const truth = rec.resNormExact / norm(b);
                worst = Math.max(worst, Math.abs(truth - res.recurrence[rec.iter - 1]) / truth);
            }
            return worst;
        };
        const right = worstGap('right');
        const left = worstGap('left');
        log('recurrence-vs-true worst relative gap, right', right);
        log('recurrence-vs-true worst relative gap, left', left);
        expect(right).toBeLessThan(1e-11); // MEASURED 1.3e-12
        expect(left).toBeGreaterThan(5e-2); // MEASURED 0.23
    });

    test('right-preconditioned application counts: 1 + 1 per iteration, +1 to recover x', () => {
        const res = gmres(S.op, P, b, { tol: 0, maxit: 5, side: 'right', knockOut: true });
        expect(res.matvecs).toBe(5);
        expect(res.precondApplies).toBe(5 + 1);
    });

    test('right + probe: one Ã⁻¹ per probe to form x̃, and the stopping iterate is the terminal probe', () => {
        const res = gmres(S.op, P, b, {
            tol: 0,
            maxit: 5,
            side: 'right',
            knockOut: true,
            probe: { exactApplyA: S.exactApplyA, uOp: U_F64, schedule: { head: 2, stride: 100 } },
        });
        expect(res.probe?.records.map((r) => r.iter)).toEqual([1, 2, 5]);
        expect(res.precondApplies).toBe(5 + 3);
    });
});

// ===========================================================================
// 6. The η probe, integrated.
// ===========================================================================

describe('η probe (spec §7.3)', () => {
    const S = synth(8, 7);
    const P = inversePrecond(S, 21);
    const b = randVec(24, 5);
    const exact = S.exactApplyA;

    test('the probed iteration list is the planned schedule, exactly, and is recorded as data', () => {
        const res = gmres(S.op, P, b, {
            tol: 0,
            maxit: 25,
            // 25 > n distinct steps: stop at breakdown is possible, so use the real budget guard.
            probe: { exactApplyA: exact, uOp: U_F64, schedule: { head: 5, stride: 4 } },
        });
        const pr: EtaProbeReport | undefined = res.probe;
        expect(pr).toBeDefined();
        expect(pr?.schedule).toEqual({ head: 5, stride: 4 });
        expect(pr?.planned).toEqual([1, 2, 3, 4, 5, 9, 13, 17, 21, 25]);
        // The realized list is a prefix of the plan plus, at most, one terminal probe.
        const iters = (pr?.records ?? []).map((r) => r.iter);
        const plannedPrefix = (pr?.planned ?? []).filter((k) => k <= res.iters);
        expect(iters.filter((_, i) => !pr?.records[i].terminal)).toEqual(plannedPrefix);
    });

    test('an unscheduled stopping iterate is still probed and flagged terminal', () => {
        const res = gmres(S.op, P, b, {
            tol: 1e-3,
            probe: { exactApplyA: exact, uOp: U_F64, schedule: { head: 2, stride: 100 } },
        });
        const recs = res.probe?.records ?? [];
        expect(res.iters).toBeGreaterThan(2);
        expect(recs.map((r) => r.iter)).toEqual([1, 2, res.iters]);
        expect(recs.map((r) => r.terminal)).toEqual([false, false, true]);
    });

    test('η at the returned x̃ matches an independent dense recomputation, denominator and all', () => {
        const res = gmres(S.op, P, b, {
            tol: 1e-3,
            probe: { exactApplyA: exact, uOp: U_F64, schedule: { head: 2, stride: 100 } },
        });
        const last = lastRecord(res);
        expect(last?.iter).toBe(res.iters);
        const normA2 = res.probe?.normA2 ?? Number.NaN;
        const r = distance(b, denseMatvec(S, res.x));
        const eta = r / (normA2 * norm(res.x) + norm(b));
        log('terminal η', eta);
        expect(Math.abs((last?.etaExact ?? 0) - eta) / eta).toBeLessThan(1e-9);
        expect(Math.abs((last?.xNorm ?? 0) - norm(res.x)) / norm(res.x)).toBe(0);
        // The clause-4 lower bound ‖A_σx̃‖/‖x̃‖, recomputed densely: a RATIO, not ‖A_σx̃‖.
        const lb = norm(denseMatvec(S, res.x)) / norm(res.x);
        expect(Math.abs((last?.lowerBound ?? 0) - lb) / lb).toBeLessThan(1e-12);
        // The fixture has power only if ‖A_σx̃‖ and ‖A_σx̃‖/‖x̃‖ are far apart (here ‖x̃‖ ≈ 0.04).
        expect(Math.abs(lb - norm(denseMatvec(S, res.x))) / lb).toBeGreaterThan(0.5);
        expect(last?.lowerBound ?? Number.POSITIVE_INFINITY).toBeLessThanOrEqual(
            res.probe?.normA2 ?? 0,
        );
        // ‖A‖₂ is the 30-iteration power estimate: within 1 % of λ_max, and never above it.
        const lamMax = spectrum(S.As, 8).lambdaMax;
        expect(normA2).toBeLessThanOrEqual(lamMax * (1 + 1e-14));
        expect(normA2).toBeGreaterThan(0.99 * lamMax);
        expect(res.probe?.normA2Matvecs).toBe(30);
    });

    test('etaMin / etaMinIter are the minimum of η_EXACT over the records (the gated value)', () => {
        const res = gmres(S.op, P, b, {
            tol: 0,
            maxit: 8,
            probe: {
                exactApplyA: perturbedExact(S),
                uOp: U_F64,
                schedule: { head: 8, stride: 1 },
            },
        });
        const recs = res.probe?.records ?? [];
        const min = Math.min(...recs.map((r) => r.etaExact));
        const minBct = Math.min(...recs.map((r) => r.etaBct));
        // The fixture has power only if the two minima differ: the 2^-30 operator floors η_exact
        // near 1e-9 while η_bct keeps falling.
        expect(minBct).toBeLessThan(min / 10);
        expect(res.probe?.etaMin).toBe(min);
        expect(recs.find((r) => r.etaExact === min)?.iter).toBe(res.probe?.etaMinIter);
    });

    test('etaTarget stops at the FIRST probe at or below it and says so', () => {
        const res = gmres(S.op, P, b, {
            tol: 0,
            probe: {
                exactApplyA: exact,
                uOp: U_F64,
                schedule: { head: 20, stride: 1 },
                etaTarget: 1e-4,
            },
        });
        const recs = res.probe?.records ?? [];
        expect(res.stop).toBe('eta-target');
        expect(res.iters).toBe(lastRecord(res)?.iter ?? -1);
        expect(lastRecord(res)?.etaExact).toBeLessThanOrEqual(1e-4);
        for (const r of recs.slice(0, -1)) expect(r.etaExact).toBeGreaterThan(1e-4);
        expect(res.probe?.etaTarget).toBe(1e-4);
        expect(res.probe?.uOp).toBe(U_F64);
    });

    test('a supplied normA2 is used verbatim and costs no power-iteration matvecs', () => {
        const res = gmres(S.op, P, b, {
            tol: 1e-3,
            probe: { exactApplyA: exact, uOp: U_F64, normA2: 123.5 },
        });
        expect(res.probe?.normA2).toBe(123.5);
        expect(res.probe?.normA2Matvecs).toBe(0);
    });

    test('the probe counters match calls actually made to each injected operator', () => {
        let bctCalls = 0;
        let exactCalls = 0;
        let precondCalls = 0;
        const spyOp: HierarchicalOperator = {
            ...S.op,
            applyA: (u, out) => {
                bctCalls++;
                S.op.applyA(u, out);
            },
        };
        const spyP = {
            n: P.n,
            apply: (R: ArrayLike<number>, out: Float64Array) => {
                precondCalls++;
                P.apply(R, out);
            },
        };
        const res = gmres(spyOp, spyP, b, {
            tol: 0,
            maxit: 7,
            probe: {
                exactApplyA: (u, out) => {
                    exactCalls++;
                    exact(u, out);
                },
                uOp: U_F64,
                schedule: { head: 4, stride: 2 },
            },
        });
        const pr = res.probe;
        expect(res.iters).toBe(7);
        expect(res.matvecs).toBe(7);
        expect(res.precondApplies).toBe(8);
        expect(pr?.planned).toEqual([1, 2, 3, 4, 6]);
        expect(pr?.records.map((r) => r.iter)).toEqual([1, 2, 3, 4, 6, 7]); // 7 = terminal
        expect(pr?.exactMatvecs).toBe(6);
        expect(pr?.bctMatvecs).toBe(6);
        expect(exactCalls).toBe(6);
        expect(bctCalls).toBe(res.matvecs + (pr?.normA2Matvecs ?? 0) + (pr?.bctMatvecs ?? 0));
        expect(precondCalls).toBe(res.precondApplies);
    });
});

describe('INVALID clauses, integrated (spec §7.3)', () => {
    const S = synth(8, 7);
    const P = inversePrecond(S, 21);
    const b = randVec(24, 5);
    const perturbed = perturbedExact(S);
    const run = (probeOverrides: Partial<EtaProbeOptions>) =>
        gmres(S.op, P, b, {
            tol: 0,
            maxit: 10,
            probe: { exactApplyA: perturbed, uOp: U_F64, ...probeOverrides },
        });
    /** The f64-floor scenario: the exact operator is the SAME matrix, rounded differently. */
    const runAtFloor = (uOp: number) =>
        gmres(S.op, P, b, {
            tol: 0,
            maxit: 10,
            probe: { exactApplyA: S.exactApplyA, uOp },
        });

    test('control: a correct f64 run with its own precision is VALID', () => {
        const res = run({});
        log('control etaMin', res.probe?.etaMin);
        expect(res.probe?.invalid).toEqual([]);
    });

    test('clause 2 is scoped to the OPERATOR precision: u_f32 on an f64 run is INVALID, u_f64 is not', () => {
        const wrong = runAtFloor(U_F32);
        log('f64 floor etaMin (same matrix, two roundings)', wrong.probe?.etaMin);
        expect(wrong.probe?.invalid).toContain('eta-below-uop');
        const right = runAtFloor(U_F64);
        expect(right.probe?.invalid).not.toContain('eta-below-uop');
    });

    test('clause 3: the exact matvec IS the iteration matvec ⇒ η_exact bit-equals η_bct ⇒ INVALID', () => {
        const res = run({ exactApplyA: S.op.applyA });
        expect(res.probe?.invalid).toContain('exact-equals-bct');
        for (const r of res.probe?.records ?? []) {
            expect(r.etaExact).toBe(r.etaBct);
            expect(r.violations).toContain('exact-equals-bct');
        }
    });

    test('clause 4: an under-estimated ‖A_σ‖₂ is caught against the free lower bound', () => {
        const lamMax = spectrum(S.As, 8).lambdaMax;
        const bad = run({ normA2: 0.01 * lamMax });
        expect(bad.probe?.invalid).toContain('norm-below-lower-bound');
        const ok = run({ normA2: lamMax });
        expect(ok.probe?.invalid).not.toContain('norm-below-lower-bound');
    });

    test('clause 1: the residual vanishes identically (n = 1, A_σ = 2I, b = e1 — all exact in f64)', () => {
        const op1: HierarchicalOperator = {
            n: 1,
            sigma: 2,
            applyA: (_u, out) => out.fill(0),
        };
        const res = gmres(op1, identityPrecond(1), Float64Array.of(1, 0, 0), {
            tol: 0,
            probe: { exactApplyA: (_u, out) => out.fill(0), uOp: U_F64 },
        });
        expect(res.x[0]).toBe(0.5);
        expect(res.recurrence[0]).toBe(0); // exactly 0 with tol = 0: still NOT a 'tol' stop
        expect(res.stop).toBe('breakdown');
        expect(res.probe?.records[0].etaExact).toBe(0);
        expect(res.probe?.records[0].violations).toContain('eta-zero');
        expect(res.probe?.invalid).toContain('eta-zero');
    });
});

// ===========================================================================
// 7. The real hierarchical operator and the shipped preconditioner.
// ===========================================================================

describe('real operator: structural counters against Preconditioner.solves', () => {
    test('left, n=64: 1 matvec + 1 Ã⁻¹ per iteration, +1 Ã⁻¹ for the rhs; solves += 6, L^{2-s} += 1 per apply', () => {
        const { op, P } = real(64);
        const s0 = P.solves;
        const l0 = P.lowOrderApplies;
        const res = gmres(op, P, randVec(192, 3), { tol: 0, maxit: 12 });
        expect(res.iters).toBe(12);
        expect(res.matvecs).toBe(12);
        expect(res.precondApplies).toBe(13);
        // precond.ts: EXACTLY 2·nb solves and 1 L^{2-s} per apply; nb = 3 coordinate blocks.
        expect(P.solves - s0).toBe(2 * 3 * 13);
        expect(P.lowOrderApplies - l0).toBe(13);
    });

    test('the probe never touches the preconditioner on the left path', () => {
        const { op, P, exactApplyA } = real(64);
        const s0 = P.solves;
        const res = gmres(op, P, randVec(192, 3), {
            tol: 0,
            maxit: 6,
            probe: { exactApplyA, uOp: U_F64, schedule: { head: 6, stride: 1 } },
        });
        expect(P.solves - s0).toBe(2 * 3 * res.precondApplies);
        expect(res.precondApplies).toBe(7);
    });
});

describe('real operator: correctness against a dense solve (n=64)', () => {
    test('left-preconditioned GMRES reaches the dense A_σ⁻¹ b to the conditioning limit', () => {
        const n = 64;
        const { H, op, P } = real(n);
        const Ab = densify(n, (u, out) => applyAScalar(H, u, out));
        const As = new Float64Array(Ab);
        for (let i = 0; i < n * n; i++) As[i] += op.sigma / n;
        const L = cholesky(As, n);
        const b = randVec(3 * n, 17);
        const star = new Float64Array(3 * n);
        const col = new Float64Array(n);
        for (let c = 0; c < 3; c++) {
            for (let i = 0; i < n; i++) col[i] = b[c * n + i];
            cholSolve(L, n, col);
            for (let i = 0; i < n; i++) star[c * n + i] = col[i];
        }
        const res = gmres(op, P, b, { tol: 1e-13 });
        const err = distance(res.x, star) / norm(star);
        log('n=64 left GMRES iters to 1e-13 recurrence', res.iters);
        log('n=64 relative error vs dense solve', err);
        expect(res.censored).toBe(false);
        // MEASURED 6.1e-14. The ceiling a priori is κ(A_σ)·u ≈ 1e-10 at n=64; GMRES with a good
        // preconditioner lands three decades under it.
        expect(err).toBeLessThan(4e-13);
    });
});

describe('real operator: the recurrence residual is optimistic (spec §5.3)', () => {
    test('left-preconditioned, stopped at recurrence 1e-4: the TRUE relative residual is larger', () => {
        const n = 240;
        const { op, P } = real(n);
        const b = randVec(3 * n, 29);
        const res = gmres(op, P, b, { tol: 1e-4 });
        const ax = new Float64Array(3 * n);
        applyAsigma(op, res.x, ax);
        const truth = distance(b, ax) / norm(b);
        const rec = res.recurrence[res.iters - 1];
        log('n=240 left: iters', res.iters);
        log('n=240 left: recurrence', rec);
        log('n=240 left: true relative residual', truth);
        log('n=240 left: true / recurrence', truth / rec);
        // Direction only; the magnitude is operator-dependent (spec §7.3 anti-trap 2).
        expect(truth).toBeGreaterThan(rec);
    });
});

describe('real operator: the recurrence residual is SIDE-DEPENDENT (spec §5.3; gmres_lab.py:230-234)', () => {
    // `'right'` solves A M⁻¹ u = b, so its recurrence residual IS the true residual in exact
    // arithmetic; `'left'` solves M⁻¹A x = M⁻¹b, so its recurrence is in the M⁻¹ norm and is NOT.
    // Asserting BOTH on the same operator and rhs is a differential check that the side switch is
    // plumbed: an implementation that ignored `side`, or put the preconditioner on the wrong side,
    // fails exactly one of the two halves. The under-report numbers of spec §7.3 anti-trap 2 are
    // LEFT-preconditioned numbers; nothing here compares a right-preconditioned run against them.
    for (const n of [64, 240]) {
        test(`n=${n}: right == true residual at roundoff, left optimistic at EVERY iterate`, () => {
            const { op, P, exactApplyA } = real(n);
            const b = randVec(3 * n, 29);
            const run = (side: 'left' | 'right') =>
                gmres(op, P, b, {
                    tol: 0,
                    maxit: 20,
                    side,
                    knockOut: side === 'right',
                    // normA2 is irrelevant to this test (η is not read), so it is supplied to skip
                    // 30 power-iteration matvecs.
                    probe: {
                        exactApplyA,
                        uOp: U_F64,
                        normA2: 1,
                        schedule: { head: 20, stride: 1 },
                    },
                });
            // Compared against the residual of the operator the ITERATION used (resNormBct): that
            // is the one the recurrence is about.
            const gaps = (res: GmresResult) =>
                (res.probe?.records ?? []).map((r) => {
                    const truth = r.resNormBct / norm(b);
                    const rec = res.recurrence[r.iter - 1];
                    return {
                        truth,
                        rec,
                        ratio: truth / rec,
                        relGap: Math.abs(truth - rec) / truth,
                    };
                });
            const right = gaps(run('right'));
            const left = gaps(run('left'));
            expect(right.length).toBe(20);
            expect(left.length).toBe(20);
            const rightWorst = Math.max(...right.map((g) => g.relGap));
            const leftMinRatio = Math.min(...left.map((g) => g.ratio));
            const leftMaxRatio = Math.max(...left.map((g) => g.ratio));
            log(`n=${n} right: worst |true - recurrence| / true over 20 iterates`, rightWorst);
            log(
                `n=${n} left: min / max (true / recurrence) over 20 iterates`,
                `${leftMinRatio} / ${leftMaxRatio}`,
            );
            // MEASURED 9.7e-12 (n=64) and 6.0e-12 (n=240); bound ~5x.
            expect(rightWorst).toBeLessThan(5e-11);
            // Optimistic at every iterate (MEASURED min ratio 1.03 at n=64, 1.31 at n=240) ...
            expect(leftMinRatio).toBeGreaterThan(1);
            // ... and by orders of magnitude somewhere (MEASURED max 3.0e2 at n=64, 8.4e3 at n=240).
            // The floor of 10 is far under both: the magnitude is operator-dependent and is not pinned.
            expect(leftMaxRatio).toBeGreaterThan(10);
        });
    }
});

describe('real operator: η_exact vs η_bct', () => {
    test('the χ=0 and θ=0.5 matvecs give different η at every probe (clause 3 does not fire)', () => {
        const { op, P, exactApplyA } = real(64);
        const res = gmres(op, P, randVec(192, 3), {
            tol: 0,
            maxit: 15,
            probe: { exactApplyA, uOp: U_F64, schedule: { head: 15, stride: 1 } },
        });
        for (const r of res.probe?.records ?? []) expect(r.etaExact).not.toBe(r.etaBct);
        log('n=64 etaMin (exact)', res.probe?.etaMin);
        expect(res.probe?.invalid).not.toContain('exact-equals-bct');
    });
});
