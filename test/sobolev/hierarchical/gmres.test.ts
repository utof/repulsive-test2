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
 *
 * Why that holds, MEASURED with the seeds of the 'Arnoldi / Givens structure' fixture (seed 7,
 * preconditioner seed 21, `b = randVec(3n, 5)`, `retainArnoldi`): on the default spectrum at n = 8,
 * `‖VᵀV−I‖` is 2.2e-16 / 5.0e-12 / 9.1e-10 at K = 1 / 6 / 7 with `hsub` = 0.41 / 0.062 / 6.4e-3,
 * and 0.92 at K = 8 = n, where the Krylov space is exhausted (`hsub` 1.2e-9, recurrence 1.5e-16):
 * the basis is then noise and no roundoff-level bar can hold (the `u/‖r_k‖` law of the
 * orthogonality test below). With `spread` `{ lo: 0.01, hi: 100 }` at n = 12, K = 6, `hsub` is 1.15
 * (recurrence 4.1e-2) and the loss is 1.26e-14, which IS a gate: a dropped `v_k` is O(1e-1).
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
 * An 'exact' operator that differs from the iteration's by a relative `rel`, the way the χ=0 matvec
 * differs from the θ matvec. Clause 3 is about a residual formed with THE SAME operator; two
 * roundings of one operator (`Synth.exactApplyA`) land on the f64 floor, where bit-equal η is a
 * coin flip rather than a bug (MEASURED: it fired in the control).
 *
 * `rel` is a knob because the two η's separate only below the perturbation: at `2^-30` they agree
 * to 3 digits down to 6e-8 and differ only at 3.6e-10 vs 1.7e-16 (MEASURED), which is under every
 * usable `τ_η`, whereas at `1e-2` `η_exact` floors at 3.86e-3 and `η_bct` keeps falling — the
 * fixture power the `etaTarget` currency test needs.
 */
function perturbedExact(
    S: Synth,
    rel = 2 ** -30,
): (u: ArrayLike<number>, out: Float64Array) => void {
    return (u, out) => {
        S.exactApplyA(u, out);
        for (let i = 0; i < out.length; i++) out[i] *= 1 + rel;
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

    // Clauses 1 and 2 govern BOTH η, independently (spec §7.3, the 2026-10-11 follow-up ruling box):
    // `η_bct` is the gated quantity and `η_exact` carries no bar, so a guard that read only
    // `η_exact` watched the quantity with no bar and left the bar itself unguarded. `base.etaExact`
    // stays healthy in every case below, so each assertion can only be moved by the `η_bct` half.
    test('clause 1 on η_bct: η_bct == 0 with a healthy η_exact is INVALID, and ONLY clause 1', () => {
        expect(v({ etaBct: 0 }, U_F32)).toEqual(['eta-zero']);
        expect(v({ etaBct: Number.MIN_VALUE }, U_F32)).toEqual(['eta-below-uop']);
    });

    test('clause 2 on η_bct: the /10, the direction, and the SCOPE to the operator precision', () => {
        const edge = U_F32 / 10;
        expect(v({ etaBct: edge * 0.99 }, U_F32)).toEqual(['eta-below-uop']);
        expect(v({ etaBct: edge * 1.01 }, U_F32)).toEqual([]);
        // The same f64 floor as the η_exact case above: a correct f64 η_bct (spec §7.3, MEASURED
        // 2.047e-16..6.001e-16) is not below u_f64/10, but is below u_f32/10.
        expect(v({ etaBct: 3e-16 }, U_F64)).toEqual([]);
        expect(v({ etaBct: 3e-16 }, U_F32)).toEqual(['eta-below-uop']);
        expect(v({ etaBct: (U_F64 / 10) * 0.5 }, U_F64)).toEqual(['eta-below-uop']);
    });

    test('the two η are checked independently, and a clause is reported ONCE however many η trip it', () => {
        // Both zero: one 'eta-zero' (clause 3 also fires, because 0 === 0 — a different clause).
        expect(v({ etaExact: 0, etaBct: 0 }, U_F32)).toEqual(['eta-zero', 'exact-equals-bct']);
        // Both below u_op/10 (5.96e-9): one 'eta-below-uop'.
        expect(v({ etaExact: 1e-9, etaBct: 2e-9 }, U_F32)).toEqual(['eta-below-uop']);
        // Disjointness is PER η: one η at 0 and the other below u_op/10 trips both clauses.
        expect(v({ etaExact: 0, etaBct: 1e-9 }, U_F32)).toEqual(['eta-zero', 'eta-below-uop']);
        expect(v({ etaExact: 1e-9, etaBct: 0 }, U_F32)).toEqual(['eta-zero', 'eta-below-uop']);
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
        const lamMax = spectrum(new Float64Array(S.As), 8).lambdaMax;
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
        // §7.4's `iters ∈ {0, 1}` INVALID predicate: legal here, and flagged anyway, because a
        // count of 1 cannot be told from "τ was above the initial η" by the count alone.
        expect(res.degenerate).toBe(true);
        // Both conditions hold at iteration 1 (recurrence 1.3e-16 <= tol 1e-13 AND breakdown), and
        // the recurrence tol takes precedence over breakdown.
        expect(res.stop).toBe('tol');
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
            // MEASURED 1.379e-15 (c=1e-20) and 2.293e-13 (c=1e+20); the bound is 8.7x the worse.
            expect(worst).toBeLessThan(2e-12);
            const xGap = distance(r.x, ref.x) / norm(ref.x);
            log(`scale ${c}: ‖x̃ − x̃_ref‖/‖x̃_ref‖`, xGap);
            // MEASURED 3.760e-16 (c=1e-20) and 3.667e-16 (c=1e+20) — scale invariance of `x̃` is
            // exact to f64 roundoff, 13x under this bound, NOT merely "small".
            expect(xGap).toBeLessThan(5e-15);
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
        //
        // This bar is a statement about WHERE THE ITERATION IS, not about MGS quality, and it must
        // not be "fixed" with reorthogonalisation. For MGS-GMRES `‖VᵀV−I‖ ≈ c·u_f64/‖r_k‖`, `c`
        // O(1)..O(10): orthogonality is lost COMPLETELY only as the recurrence residual reaches
        // roundoff (Greenbaum-Rozložník-Strakoš, BIT 37(3), 1997; Paige-Rozložník-Strakoš, SIAM J.
        // Matrix Anal. Appl. 28(1), 2006), and MGS-GMRES is backward stable all the same. So once
        // the recurrence has reached ~5e-14 (K = 38 below) the loss is O(1e-3..1e-2) and EXPECTED,
        // not a defect. MEASURED on `real(64)` (RS-preconditioned), `b = randVec(192, 3)`,
        // `tol = 0`, `maxit = K`:
        //     K    ‖VᵀV−I‖    recurrence   u_f64/rec   ratio
        //    10   1.108e-12    3.214e-4    3.454e-13    3.21
        //    20   2.756e-10    1.469e-6    7.557e-11    3.65
        //    30   3.164e-06    1.309e-10   8.482e-07    3.73
        //    38   8.131e-03    4.809e-14   2.309e-03    3.52
        //    50   6.384e-01    4.667e-16   2.379e-01    2.68
        // — a flat prefactor over ten decades (1.06..1.49 on `randVec(192, 17)`; it depends on `b`,
        // not on K). The consequence: at K = 50, with the basis essentially gone, `η_bct` is
        // 6.8e-16 (≈ 6 u_f64; 4.8e-16 on seed 17), i.e. backward stable. A second orthogonalisation
        // pass would buy nothing and cost a pass per step.
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
        // The smallest count that is NOT §7.4's `iters ∈ {0, 1}`: a `degenerate` that fired at 2
        // would mark this healthy exact-termination run INVALID (MEASURED iters = 2, stop 'tol').
        expect(res.iters).toBe(2);
        expect(res.degenerate).toBe(false);
    });

    test('happy breakdown: an exact inverse exhausts the Krylov space at iteration 1', () => {
        const S = synth(8, 7);
        const res = solveSynth(S, inversePrecond(S, 7), randVec(24, 11), { tol: 0 });
        expect(res.iters).toBe(1);
        expect(res.stop).toBe('breakdown');
        expect(res.censored).toBe(false);
        expect(res.degenerate).toBe(true);
    });

    test('stop precedence: η target over recurrence tol over breakdown, when all three hold at once', () => {
        // An exact inverse makes iteration 1 satisfy every stop rule simultaneously: the recurrence
        // is 1.3e-16 <= tol, `hsub/‖Mv‖` is 2.2e-16 <= BREAKDOWN_REL, and η is at the f64 floor.
        // Only the reported label distinguishes the orderings, which is why it needs its own test.
        const S = synth(8, 7);
        const exact = inversePrecond(S, 7);
        const b = randVec(24, 11);
        const all = gmres(S.op, exact, b, {
            tol: 1e-13,
            probe: {
                exactApplyA: S.exactApplyA,
                uOp: U_F64,
                schedule: { head: 4, stride: 1 },
                etaTarget: 1e-3,
            },
        });
        expect(all.iters).toBe(1);
        expect(all.stop).toBe('eta-target');
        // Without the η target, `tol` wins over breakdown; with neither, breakdown is reported.
        expect(solveSynth(S, exact, b, { tol: 1e-13 }).stop).toBe('tol');
        expect(solveSynth(S, exact, b, { tol: 0 }).stop).toBe('breakdown');
    });

    test('on breakdown the REPORTED basis is not extended with the noise vector', () => {
        // The `!breakdown` guard before pushing `V[k+1]` cannot change the iteration (breakdown is
        // a stop condition and `solution()` reads `V[0..k-1]`), so `arnoldi.basis.length` is the
        // only place it is observable — which is exactly what the guard's comment claims.
        const S = synth(8, 7);
        const b = randVec(24, 11);
        const broke = solveSynth(S, inversePrecond(S, 7), b, { tol: 0, retainArnoldi: true });
        expect(broke.stop).toBe('breakdown');
        expect(broke.arnoldi?.basis.length).toBe(broke.iters); // 1, NOT 2
        const ran = solveSynth(S, inversePrecond(S, 21), b, {
            tol: 0,
            maxit: 5,
            retainArnoldi: true,
        });
        expect(ran.stop).toBe('maxit');
        expect(ran.arnoldi?.basis.length).toBe(ran.iters + 1);
    });

    test('the breakdown threshold is below a GRADED subdiagonal that is still a real Krylov direction', () => {
        // `Ã⁻¹ = D A_σ⁻¹` with `D = diag(1 ± 1e-13)`: the iteration-1 Arnoldi subdiagonal is a
        // chosen 1e-13 of the pre-orthogonalisation norm instead of the exact inverse's 2.2e-16.
        // That pins BREAKDOWN_REL's UPPER side, which is otherwise unconstrained: a threshold at or
        // above this ratio truncates the solve at iteration 1 (MEASURED: `iters` 1, `degenerate`).
        const S = synth(8, 7);
        const exact = inversePrecond(S, 7);
        const graded = {
            n: 8,
            apply: (R: ArrayLike<number>, out: Float64Array): void => {
                exact.apply(R, out);
                for (let i = 0; i < out.length; i++) out[i] *= 1 + (i % 2 === 0 ? 1e-13 : -1e-13);
            },
        };
        const res = solveSynth(S, graded, randVec(24, 11), {
            tol: 0,
            maxit: 6,
            retainArnoldi: true,
        });
        const h0 = res.arnoldi?.hessenberg[0] ?? new Float64Array(2);
        const ratio = h0[1] / Math.hypot(h0[0], h0[1]);
        log('graded fixture: iteration-1 hsub / ‖M v₀‖', ratio);
        // MEASURED 9.864e-14, against 2.181e-16 for the same preconditioner un-graded (452x).
        expect(ratio).toBeGreaterThan(5e-14);
        expect(ratio).toBeLessThan(2e-13);
        // ... and at the shipped 1e-14 this is NOT a breakdown, so the iteration runs its budget.
        expect(res.iters).toBe(6);
        expect(res.degenerate).toBe(false);
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
        expect(res.degenerate).toBe(true); // §7.4's `iters ∈ {0, 1}`: the other end of it
    });

    test('degenerate is §7.4s `iters ∈ {0,1}` and is FALSE on a healthy run (censored is the sibling)', () => {
        const S = synth(8, 7);
        const b = randVec(24, 11);
        const healthy = solveSynth(S, inversePrecond(S, 21), b, { tol: 1e-6 });
        expect(healthy.iters).toBe(7);
        expect(healthy.degenerate).toBe(false);
        expect(healthy.censored).toBe(false);
        // The two predicates are independent: a 1-iteration budget is both.
        const capped = solveSynth(S, inversePrecond(S, 21), b, { tol: 0, maxit: 1 });
        expect(capped.iters).toBe(1);
        expect(capped.degenerate).toBe(true);
        expect(capped.censored).toBe(true);
        // ... and a run that is censored at a real count is NOT degenerate.
        const long = solveSynth(S, identityPrecond(8), b, { tol: 0, maxit: 3 });
        expect(long.censored).toBe(true);
        expect(long.degenerate).toBe(false);
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
        // MEASURED 1.146e-14: the probe's η and a dense recomputation of the same formula differ
        // only by the two summation orders. The bound is 8.7x that, so a multiplicative error in
        // either denominator term cannot hide here (it was 0.956 for a swapped-argument mutant).
        expect(Math.abs((last?.etaExact ?? 0) - eta) / eta).toBeLessThan(1e-13);
        expect(Math.abs((last?.xNorm ?? 0) - norm(res.x)) / norm(res.x)).toBe(0);
        // The clause-4 lower bound ‖A_σx̃‖/‖x̃‖, recomputed densely: a RATIO, not ‖A_σx̃‖.
        const lb = norm(denseMatvec(S, res.x)) / norm(res.x);
        // MEASURED exactly 0 — bit-identical. The allowance is ONE f64 eps and not a round number,
        // because the two routes sum in different orders (`Synth.exactApplyA` accumulates over
        // columns, `denseMatvec` over rows) so bit-equality is not structural.
        expect(Math.abs((last?.lowerBound ?? 0) - lb) / lb).toBeLessThanOrEqual(2 ** -52);
        // The fixture has power only if ‖A_σx̃‖ and ‖A_σx̃‖/‖x̃‖ are far apart (here ‖x̃‖ ≈ 0.04).
        expect(Math.abs(lb - norm(denseMatvec(S, res.x))) / lb).toBeGreaterThan(0.5);
        expect(last?.lowerBound ?? Number.POSITIVE_INFINITY).toBeLessThanOrEqual(
            res.probe?.normA2 ?? 0,
        );
        // ‖A‖₂ is the 30-iteration power estimate, and it is a LOWER bound that tightens.
        // `spectrum` DESTROYS its argument (`denseSpectrum.ts` `tridiagonalize`), so it gets a copy:
        // `S` is shared by this whole describe block and `S.As` is the oracle of other tests in it.
        const lamMax = spectrum(new Float64Array(S.As), 8).lambdaMax;
        expect(normA2).toBeLessThanOrEqual(lamMax * (1 + 1e-14));
        // MEASURED ratio 0.998790, i.e. a shortfall of 1.21e-3; the bound allows 2.5x that. The
        // shortfall is the quantity with a floor (the near-degenerate top eigenvalues, spec §7.3),
        // so it is what carries the margin, not the ratio.
        expect(normA2).toBeGreaterThan(0.997 * lamMax);
        expect(res.probe?.normA2Matvecs).toBe(30);
    });

    test('etaMin is the minimum of η_EXACT (the δ_compress monitor) and etaBctMin of η_BCT (H2s gated value)', () => {
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
        // Both currencies are reported, and they are NOT the same number — spec §7.3's amendment
        // box makes η_bct the gated one and leaves η_exact as the δ_compress monitor.
        expect(res.probe?.etaBctMin).toBe(minBct);
        expect(recs.find((r) => r.etaBct === minBct)?.iter).toBe(res.probe?.etaBctMinIter);
        expect(res.probe?.etaBctMin).not.toBe(res.probe?.etaMin);
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
        expect(lastRecord(res)?.etaBct).toBeLessThanOrEqual(1e-4);
        for (const r of recs.slice(0, -1)) expect(r.etaBct).toBeGreaterThan(1e-4);
        expect(res.probe?.etaTarget).toBe(1e-4);
        expect(res.probe?.uOp).toBe(U_F64);
    });

    // The currency of the stop is η_BCT, per the 2026-10-11 amendment box in spec §7.3 ("the gated
    // quantity is `η_bct`") and §7.4 ("H3 counts on `η_bct`"). The fixture above cannot see which
    // η is read — on an unperturbed exact operator the two cross 1e-4 at the same iteration 6
    // (MEASURED) — so the currency needs a fixture where only one of them is reachable.
    describe('etaTarget reads η_BCT, not η_exact (spec §7.3 / §7.4, the 2026-10-11 amendment)', () => {
        // relative 1e-2: η_exact floors at 3.856e-3 at iteration 5 and never falls again, so
        // NO target below it is reachable on η_exact; η_bct passes 1e-4 at iteration 6 and
        // 2.176e-4 at iteration 5 (MEASURED, all five seeds of the fixture are deterministic).
        const coarse = perturbedExact(S, 1e-2);
        const runTarget = (etaTarget?: number) =>
            gmres(S.op, P, b, {
                tol: 0,
                maxit: 20,
                probe: {
                    exactApplyA: coarse,
                    uOp: U_F64,
                    schedule: { head: 20, stride: 1 },
                    etaTarget,
                },
            });

        test('a target unreachable on η_exact but reachable on η_bct DOES stop the solve', () => {
            const res = runTarget(1e-4);
            // Reading η_exact here gives `stop: 'maxit'` at `iters: 20` (MEASURED on the mutant).
            expect(res.stop).toBe('eta-target');
            expect(res.iters).toBe(6);
            const recs = res.probe?.records ?? [];
            // The fixture has power only if the target is out of η_exact's reach at EVERY iterate.
            expect(res.probe?.etaMin).toBeGreaterThan(1e-4);
            expect(Math.min(...recs.map((r) => r.etaExact))).toBeGreaterThan(1e-4);
            expect(lastRecord(res)?.etaBct).toBeLessThanOrEqual(1e-4);
            // ... and η_exact at the stopping iterate is four decades above it (MEASURED 3.86e-3).
            expect(lastRecord(res)?.etaExact).toBeGreaterThan(1e-3);
        });

        test('the etaTarget comparison is INCLUSIVE, on η_bct: the iteration-5 value stops at 5', () => {
            const recs = runTarget().probe?.records ?? [];
            const edge = recs[4].etaBct; // MEASURED 2.175739e-4
            expect(edge).toBeGreaterThan(0);
            const res = runTarget(edge);
            expect(res.iters).toBe(5);
            expect(res.stop).toBe('eta-target');
            // A strict `<` would carry on to iteration 6 (MEASURED on the mutant).
            expect(lastRecord(res)?.etaBct).toBe(edge);
        });
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

    test('clause 1 on η_bct: the gated residual vanishes while η_exact does not (n = 1, A_σ = 2I)', () => {
        // Same fixture as the η_exact clause-1 test below, but the "exact" operator is `0.25·I` off
        // the iteration's, so x̃ = 0.5 e1 satisfies A_σ x̃ = b EXACTLY while the χ=0 residual is
        // 0.125 (η_exact = 0.05 with the supplied ‖A_σ‖₂ = 3, which also clears the free lower
        // bound 2.25 of clause 4). Only the `η_bct` half of the clause can see this.
        const op1: HierarchicalOperator = {
            n: 1,
            sigma: 2,
            applyA: (_u, out) => out.fill(0),
        };
        const res = gmres(op1, identityPrecond(1), Float64Array.of(1, 0, 0), {
            tol: 0,
            probe: {
                exactApplyA: (u, out) => {
                    for (let i = 0; i < 3; i++) out[i] = 0.25 * u[i];
                },
                uOp: U_F64,
                normA2: 3,
            },
        });
        const rec = res.probe?.records[0];
        expect(rec?.etaBct).toBe(0);
        // The fixture precondition: η_exact is healthy, so the narrow predicate reads this VALID.
        expect(rec?.etaExact).toBeGreaterThan(U_F64 / 10);
        expect(rec?.violations).toEqual(['eta-zero']);
        expect(res.probe?.invalid).toEqual(['eta-zero']);
    });

    test('clause 2 on η_bct: η_bct under u_op/10 while η_exact is not (hand-supplied u_op = 1e-9)', () => {
        // `perturbed` floors η_exact at 3.6e-10 and η_bct falls to ~1.7e-16 by iteration 8
        // (MEASURED, the 10 records of `run({})`). A u_op of 1e-9 puts the bar u_op/10 = 1e-10
        // between them: 3.6x under the η_exact floor, ~6 decades over η_bct. A hand-supplied u_op
        // is legitimate because `etaViolations` takes it as a parameter; no real operator has it.
        const uOp = 1e-9;
        const res = run({ uOp });
        const records = res.probe?.records ?? [];
        // The fixture precondition: no η_exact is under the bar, so the narrow predicate reads
        // every record VALID and only the `η_bct` half can fire.
        for (const r of records) expect(r.etaExact).toBeGreaterThanOrEqual(uOp / 10);
        const tripped = records.filter((r) => r.violations.includes('eta-below-uop'));
        expect(tripped.map((r) => r.iter)).toEqual([8, 9, 10]);
        for (const r of tripped) expect(r.etaBct).toBeLessThan(uOp / 10);
        expect(res.probe?.invalid).toEqual(['eta-below-uop']);
    });

    test('clause 4: an under-estimated ‖A_σ‖₂ is caught against the free lower bound', () => {
        const lamMax = spectrum(new Float64Array(S.As), 8).lambdaMax;
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
    // STOPPED ON THE η PROBE, NOT ON THE RECURRENCE RESIDUAL, and that is the whole design of this
    // test. Left-preconditioned the recurrence is `‖Ã⁻¹r‖/‖Ã⁻¹b‖` (spec §5.3), so a run stopped at
    // `tol = 1e-13` stops at a backward error that scales with `κ(Ã⁻¹A_σ)` — the quantity the mass
    // weight moves (@issue utof/repulsive-test2#72) — and the forward error inherits it. MEASURED
    // on exactly that rule: `err` = 9.34e-15 / 6.15e-14 / 1.50e-12 / 2.83e-11 at
    // `massWeight` = 10 / 1 / 0.1 / 0.001, a 3 000x spread over four decades of a parameter still
    // under adjudication. Stopping on `η_bct <= 1e-10` instead FIXES the backward error, so the
    // same sweep gives `err` = 1.28e-10 / 8.33e-11 / 9.97e-11 / 8.24e-11 — a 1.55x spread — and
    // every bound below is a bound on a quantity the preconditioner cannot move.
    test('left-preconditioned GMRES reaches the dense A_σ⁻¹ b at the backward error it stopped on', () => {
        const n = 64;
        const { H, op, P, exactApplyA } = real(n);
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
        const ETA_TARGET = 1e-10; // reachable at every massWeight in 1e-3..10 (MEASURED floors 3e-16..4e-12)
        const res = gmres(op, P, b, {
            tol: 0,
            maxit: 60,
            probe: {
                exactApplyA,
                uOp: U_F64,
                schedule: { head: 60, stride: 1 },
                etaTarget: ETA_TARGET,
            },
        });
        const err = distance(res.x, star) / norm(star);
        // The true relative residual against the DENSIFIED A_σ this test already builds — the
        // same quantity η is formed from, without η's `‖A‖‖x̃‖` denominator term.
        const ax = new Float64Array(3 * n);
        for (let c = 0; c < 3; c++) {
            for (let i = 0; i < n; i++) {
                let acc = 0;
                for (let j = 0; j < n; j++) acc += As[i * n + j] * res.x[c * n + j];
                ax[c * n + i] = acc;
            }
        }
        const trueRes = distance(b, ax) / norm(b);
        log('n=64 left GMRES iters to η_bct 1e-10', res.iters);
        log('n=64 relative error vs dense solve', err);
        log('n=64 true relative residual vs the densified A_σ', trueRes);
        log('n=64 min η_exact over the probes (the δ_compress monitor)', res.probe?.etaMin);
        expect(res.stop).toBe('eta-target');
        expect(res.censored).toBe(false);
        expect(res.degenerate).toBe(false);
        expect(res.probe?.etaBctMin).toBeLessThanOrEqual(ETA_TARGET);
        // MEASURED 8.33e-11 at the shipped massWeight and 1.28e-10 worst over 1e-3..10; bound 3.9x
        // the worst. The a-priori ceiling is `κ₂(A_σ)·η·(‖A‖‖x̃‖+‖b‖)/‖b‖` = 200 · 1e-10 · 41 =
        // 8e-7 (Rigal-Gaches), so this lands 4 decades under what the conditioning allows.
        expect(err).toBeLessThan(5e-10);
        // MEASURED 2.54e-9 at the shipped massWeight, 3.43e-9 worst over the same sweep; bound 2.9x.
        expect(trueRes).toBeLessThan(1e-8);
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
    // `'right'` solves A M⁻¹ u = b, so its recurrence residual IS `‖r‖/‖b‖` in exact arithmetic;
    // `'left'` solves M⁻¹A x = M⁻¹b, so its recurrence is `‖Ã⁻¹r‖/‖Ã⁻¹b‖` and is NOT. What is
    // asserted is that EXACT ALGEBRAIC IDENTITY per side, plus the CROSS term — each side's
    // recurrence against the other side's quantity — which is what makes this a differential: an
    // implementation that ignored `side`, or applied `Ã⁻¹` on the wrong side, fails one half by
    // O(1) (MEASURED: a side-ignoring mutant reads 0.23 and 0.997 here).
    //
    // WHAT IS DELIBERATELY *NOT* ASSERTED, AND WHY — do not re-promote it. The direction of the
    // left gap REVERSES with the preconditioner's mass weight (@issue utof/repulsive-test2#72).
    // MEASURED `min / max (true / recurrence)` over 20 iterates:
    //   massWeight 10    : 0.8459 / 1.217   (n=64)   0.6447 / 0.8715  (n=240)  ⇐ PESSIMISTIC
    //   massWeight 1     : 1.0277 / 301.6            1.3136 / 8 423
    //   massWeight 0.1   : 4.299  / 1.221e4          5.930  / 1.133e5
    //   massWeight 0.001 : 16.77  / 9.242e4          16.64  / 5.630e5
    // So "left is optimistic at every iterate" is true at the shipped weight with a 2.8 % margin
    // and FALSE at massWeight 10. The ratios are RECORDED below instead, and the identity — which
    // is exact algebra — carries the gate. §5.3's "it under-reports" is a statement about the
    // shipped configuration, not about the preconditioner family.
    const HEAD = 3;
    const K = 20;
    for (const n of [64, 240]) {
        test(`n=${n}: the recurrence IS its own side's residual identity, and NOT the other side's`, () => {
            const { op, P, exactApplyA } = real(n);
            const b = randVec(3 * n, 29);
            const bNorm = norm(b);
            const pb = new Float64Array(3 * n);
            P.apply(b, pb);
            const betaLeft = norm(pb);
            const run = (side: 'left' | 'right', maxit = K, probe = true) =>
                gmres(op, P, b, {
                    tol: 0,
                    maxit,
                    side,
                    knockOut: side === 'right',
                    // normA2 is irrelevant to this test (η is not read), so it is supplied to skip
                    // 30 power-iteration matvecs.
                    probe: probe
                        ? { exactApplyA, uOp: U_F64, normA2: 1, schedule: { head: K, stride: 1 } }
                        : undefined,
                });
            // `‖r_k‖/‖b‖` at every iterate comes free from the probe (`resNormBct` is the residual
            // of the operator the ITERATION applied, recomputed from `b`).
            const trueRel = (res: GmresResult) =>
                (res.probe?.records ?? []).map((r) => r.resNormBct / bNorm);
            // `‖Ã⁻¹r_k‖/‖Ã⁻¹b‖` needs `r_k` itself, so `x̃_k` is recomputed by re-running to
            // `maxit = k` — only over the HEAD window, which is all the identity needs.
            const precondRel = (side: 'left' | 'right'): number[] => {
                const ax = new Float64Array(3 * n);
                const rv = new Float64Array(3 * n);
                const pr = new Float64Array(3 * n);
                const out: number[] = [];
                for (let k = 1; k <= HEAD; k++) {
                    const sub = run(side, k, false);
                    applyAsigma(op, sub.x, ax);
                    for (let i = 0; i < 3 * n; i++) rv[i] = b[i] - ax[i];
                    P.apply(rv, pr);
                    out.push(norm(pr) / betaLeft);
                }
                return out;
            };
            const relGap = (q: number, rec: number) => Math.abs(q - rec) / q;
            const worst = (xs: number[], rec: number[]) =>
                Math.max(...xs.map((q, i) => relGap(q, rec[i])));

            const right = run('right');
            const left = run('left');
            const recRight = Array.from(right.recurrence);
            const recLeft = Array.from(left.recurrence);
            expect(trueRel(right).length).toBe(K);
            expect(trueRel(left).length).toBe(K);

            // --- each side against its OWN identity, on the head window -------------------------
            const ownRight = worst(trueRel(right).slice(0, HEAD), recRight);
            const ownLeft = worst(precondRel('left'), recLeft);
            log(
                `n=${n} identity defect over iterates 1..${HEAD}, right / left`,
                `${ownRight} / ${ownLeft}`,
            );
            // MEASURED worst over massWeight ∈ {10, 1, 0.1, 0.001} × n ∈ {64, 240}: 5.768e-14
            // (at the shipped weight: 1.07e-15 / 9.45e-16 at n=64, 5.77e-14 / 1.13e-15 at n=240).
            // Bound 8.7x the worst — the identity holds to f64 roundoff whatever the mass weight,
            // which is why THIS is the gate and the 20-iterate figure below is not.
            expect(ownRight).toBeLessThan(5e-13);
            expect(ownLeft).toBeLessThan(5e-13);

            // --- each side against the OTHER side's quantity: the differential ------------------
            const crossRight = worst(precondRel('right'), recRight);
            const crossLeft = worst(trueRel(left).slice(0, HEAD), recLeft);
            log(
                `n=${n} CROSS defect over iterates 1..${HEAD}, right / left`,
                `${crossRight} / ${crossLeft}`,
            );
            // MEASURED min over the same 8 configurations 1.198e-1, i.e. the two quantities are
            // never within 12 % of each other; bound 2.4x under that. A one-sided test is what
            // lets a side-plumbing defect through, so both halves are here.
            expect(crossRight).toBeGreaterThan(0.05);
            expect(crossLeft).toBeGreaterThan(0.05);

            // --- the 20-iterate figure: RECORDED, with a bound that survives the sweep ----------
            const rightWorst = worst(trueRel(right), recRight);
            const ratios = trueRel(left).map((q, i) => q / recLeft[i]);
            log(`n=${n} right: worst |true - recurrence| / true over ${K} iterates`, rightWorst);
            log(
                `n=${n} left: min / max (true / recurrence) over ${K} iterates — RECORDED, not gated`,
                `${Math.min(...ratios)} / ${Math.max(...ratios)}`,
            );
            // This one really does degrade with the conditioning — it is the MGS-GMRES `u·κ`
            // accumulation (Paige-Rozloznik-Strakos), not a property of the identity. MEASURED
            // 9.65e-12 (n=64) and 5.97e-12 (n=240) at the shipped weight but 1.09e-11 and 1.32e-9
            // at massWeight 0.001; the bound is 7.6x the WORST of the sweep, and it is therefore
            // NOT a tight bound at the shipped weight. Do not read it as one.
            expect(rightWorst).toBeLessThan(1e-8);
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
