/**
 * Left-preconditioned GMRES for `A_σ x = b` on the hierarchical operator (Phase 2H), with the
 * backward-error `η` probe that spec §7.3 (H2) and §7.4 (H3) are both measured with.
 *
 * `A_σ = A_bct + (σ/n) 1 1ᵀ` PER COORDINATE BLOCK (spec §2.3): `HierarchicalOperator.applyA` is
 * `A_bct` ONLY, so {@link applyAsigma} forms the rank-1 shift here, from the same `op.sigma` the
 * preconditioner was built with. The preconditioner is INJECTED and this file neither constructs
 * one nor knows its mass weight — `Ã⁻¹` is anything with `apply(R, out)` on the `3n` coordinate-block
 * layout `R[c*n + i]` (`precond.ts`). That keeps everything below valid whatever the shipped mass
 * weight turns out to be (spec amendment A14, @issue utof/repulsive-test2#72).
 *
 * ALGORITHM: full (non-restarted) GMRES, `x0 = 0`, MODIFIED Gram-Schmidt Arnoldi, Givens rotations,
 * ALL IN f64 — exactly `gmres_lab.py:217-300`'s `gmres`, the code that measured every floor in spec
 * §7.3/§7.4. Spec §7.10: "a floor measured in one precision configuration is not transferable to
 * another … if orthogonalisation moves to f32, H2's bar must be re-derived. NOT MEASURED." So the
 * orthogonalisation stays f64 and a mixed-precision inner apply (spec §6) changes only what the
 * injected `op.applyA` does. The licence for an f32 inner / f64 outer scheme is H1-b's measured
 * `δ_precision` plus the f64 outer residual, under inexact-Krylov theory (Simoncini & Szyld 2003;
 * van den Eshof & Sleijpen 2004) — NOT `[CH18]` (Carson & Higham 2018), which spec §7.8 retracts FOR
 * THAT LICENCE. The retraction does not reach the `η` FORMULA: its Rigal-Gaches (1967) / Higham
 * *ASNA* Thm 7.1 citation ({@link backwardError}) is correct and stays.
 *
 * LEFT PRECONDITIONING IS PRODUCTION; RIGHT IS A KNOCK-OUT INSTRUMENT (spec §5.3, §11.2 row 7).
 * Right-preconditioned f32 gave `cos∠_A = 0.142264` at n=480 WHILE ITS RECURRENCE RESIDUAL READ A
 * HEALTHY 1e-4: the direction is destroyed and the solver's own number says it succeeded. So
 * `side: 'right'` is REFUSED unless `knockOut: true`, mirroring `makePreconditioner`
 * (`precond.ts:272-277`) and `prepareA` (`apply.ts:482-490`): a configuration the code itself
 * calls a knock-out must not be production-reachable.
 *
 * THE RECURRENCE RESIDUAL IS NOT A CONVERGENCE CRITERION (spec §5.3). Left-preconditioned it is
 * `‖Ã⁻¹r‖/‖Ã⁻¹b‖`, and it under-reports the true relative residual by 4.372e8 / 1.404e9 /
 * 1.142e10 / 2.056e11 / 1.717e12 at n = 64…960 (spec §7.3 anti-trap 2, MEASURED; fitted exponent
 * +3.056, tracking `κ₂(A_σ) ∝ n^3.334`). It is exposed (`recurrence`) because the Givens
 * bookkeeping produces it and §5.6 requires it, and it is the `tol` stopping rule because that is
 * what GMRES can compute for free — but every convergence claim must form the residual from `b`
 * against the real operator, which is what the `η` probe does.
 *
 * COUNTS ARE STRUCTURAL INTEGERS (`matvecs`, `precondApplies`), for the reason `precond.ts:118-124`
 * gives for its own counters: a fourth `Ã⁻¹` apply per iteration (or a dense route) would read
 * exactly like a passing test without them, and wall-clock cannot settle spec amendment A9
 * because it is load-dependent (MEASURED: the same cost ratio read 1.12-1.20 under suite load vs
 * 2.27-2.54 isolated).
 *
 * `maxit` IS RECORDED and censoring is `iters === maxit`, compared against `maxit` and NEVER
 * against `n` (spec §5.3:725, §7.4's INVALID predicate, §11.2 row 7).
 *
 * NOT RE-ENTRANT, because `HierarchicalOperator` and `Preconditioner` are not.
 *
 * @see https://arxiv.org/abs/2107.01664 §5.2 (the preconditioner this is run with)
 * @see https://doi.org/10.1137/S1064827502406415 (Simoncini & Szyld, SIAM J. Sci. Comput. 25(2), 2003)
 * @see https://doi.org/10.1137/S0895479802403459 (van den Eshof & Sleijpen, SIAM J. Matrix Anal. Appl. 26(1), 2004)
 * @see .superpowers/sdd/2026-09-11-phase2h-queue/scratch/gmres_lab.py:217-300
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §2.3, §5.3, §5.6, §7.3, §7.4, §7.10, §11.2 task 7
 * @issue utof/repulsive-test2#59
 */
import type { HierarchicalOperator } from './operator';
import type { Preconditioner } from './precond';

/**
 * Unit roundoff of f32, `2^-24 = 5.96e-8` — the `u_f32` of spec §7.3's bar `50·u_f32 = 2.98e-6`.
 * Defined here rather than imported from `src/gpu/wgsl.ts` (`F32_U`) because `src/core` must not
 * depend on `src/gpu`.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.3
 */
export const U_F32 = 2 ** -24;

/**
 * Unit roundoff of f64, `2^-53 = 1.11e-16` (NOT the machine epsilon `2^-52`). Why: INVALID clause 2
 * is `η < u_op/10` with `u_op` the OPERATOR's own roundoff, and a correct f64 run floors at
 * `etaFloorMin` 2.047e-16…6.001e-16 (spec §7.3, MEASURED) — above `u_f64/10 = 1.1e-17`, below
 * `u_f32/10`.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.3
 */
export const U_F64 = 2 ** -53;

/**
 * `maxit = min(600, 2·dim)` — "exactly E1's budget, so the measured floors transfer literally"
 * (spec §7.3). `dim = 3n`, the size of the system THIS file solves.
 *
 * Do not "fix" `3n` to `n` on seeing the lab: `e1b_left_f64.py:90` writes `maxit = min(600, 2 * n)`
 * and looks like a contradiction, but the lab solves the SCALAR `n x n` system (`gmres_lab.py`
 * `load_system` returns the scalar `A_σ`; its `Minv` is `n x n`), so the lab's `dim` IS `n` and the
 * formula `min(600, 2·dim)` is the same one. The two budgets coincide for `n >= 300` (the 600 cap
 * binds); where they differ (`n < 300`) both are far above the iteration at which the `η` floor is
 * located (`etaArgminIter` 36-60, MEASURED, spec §7.3), so no measured floor moves.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.3
 * @see .superpowers/sdd/2026-09-11-phase2h-queue/scratch/e1b_left_f64.py:90
 */
export function defaultMaxit(dim: number): number {
    return Math.min(600, 2 * dim);
}

/** Power-iteration count of `‖A_σ‖₂`, spec §7.3 ("30 power iterations on the hierarchical matvec"). */
const NORM_POWER_ITERS = 30;
/** Seed of the deterministic start vector of {@link estimateNorm2}; no `Math.random` in a gate. */
const NORM_SEED = 20261009;
/**
 * Happy-breakdown threshold: the new Arnoldi vector lost all but `~1e-14` of the norm it had
 * before orthogonalisation, i.e. it lies in the span within roundoff. Normalising what is left
 * would turn rounding noise into a basis vector that is NOT orthogonal to the others. The lab
 * tests `hsub == 0.0` exactly (`gmres_lab.py:270`); an exact inverse as `Ã⁻¹` leaves `hsub ~ 1e-16`
 * here, not 0 (`gmres.test.ts` pins the iteration-1 breakdown), so the threshold is relative to the
 * norm `‖M v‖` the vector had BEFORE orthogonalisation: scaling `Ã⁻¹` by 1e-20 must not change it.
 */
const BREAKDOWN_REL = 1e-14;

/** `y = A_σ u` per coordinate block, `A_σ = A_bct + (σ/n) 1 1ᵀ` — see {@link applyAsigma}. */
type Matvec = (u: Float64Array, out: Float64Array) => void;

/**
 * `out = A_σ u` on `3n` in the coordinate-block layout: `A_bct u` plus `σ · mean(u_c)` added to
 * every entry of block `c` (spec §2.3's `(σ/n) 1 1ᵀ`, per block — `A` is block-diagonal over the
 * three components, `layout.ts` `expandBlockDiag`, so a shift across all `3n` entries would couple
 * the components).
 *
 * The scale is `σ · (Σu / n)`, in THAT order, and it is load-bearing: for `u = 1` the sum is
 * exactly `n`, `Σu / n` is exactly 1, and with `A_bct 1 = 0` BIT-EXACTLY (`apply.ts` module
 * TSDoc) the result is exactly `σ` — i.e. `A_σ 1 = σ 1` with no rounding, which is the premise of
 * `precond.ts`'s "constant-mode eigenvalue exactly σ". `(σ/n) · Σu` would round differently. The
 * sum is plain (not Neumaier): its O(n·u) error is on an O(σ) term, inside the operator's own
 * roundoff, unlike `precond.ts`'s `project`, whose structural zeros need the compensated sum.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §2.3
 */
export function applyAsigma(
    op: HierarchicalOperator,
    u: ArrayLike<number>,
    out: Float64Array,
): void {
    shiftedApply(op.applyA, op.n, op.sigma, u, out);
}

/**
 * `applyA` plus the per-block rank-1 shift. ONE copy of the shift, used for the iteration's BCT
 * matvec AND the probe's χ=0 exact matvec, so the two differ only in `applyA` — a second copy of
 * the formula could drift and would turn `η_exact` into a measure of the shift mismatch.
 */
function shiftedApply(
    applyA: (u: ArrayLike<number>, out: Float64Array) => void,
    n: number,
    sigma: number,
    u: ArrayLike<number>,
    out: Float64Array,
): void {
    applyA(u, out);
    for (let c = 0; c < 3; c++) {
        const o = c * n;
        let s = 0;
        for (let i = 0; i < n; i++) s += u[o + i];
        const shift = sigma * (s / n);
        for (let i = 0; i < n; i++) out[o + i] += shift;
    }
}

/**
 * Rigal-Gaches normwise relative backward error `η = ‖r‖ / (‖A‖‖x‖ + ‖b‖)` (1967; Higham, *ASNA*
 * Thm 7.1), given the four norms. BOTH denominator terms matter: `‖A‖‖x‖` alone is not defined for
 * `x = 0` and `‖b‖` alone is the relative residual, whose floor degrades as `n^3.36`
 * (spec §7.4) where `η`'s grows as `n^0.34`.
 *
 * It is NOT `structuredSaddleResidual` (`linsolve.ts:860`): that one's `max(1, ‖rhs‖)` denominator
 * is not relative for `‖rhs‖ < 1`, it sums the constraint rows, and it needs the dense matrix
 * (spec §7.3 anti-trap 1). Do not reuse it.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.3
 */
export function backwardError(
    resNorm: number,
    normA2: number,
    xNorm: number,
    bNorm: number,
): number {
    return resNorm / (normA2 * xNorm + bNorm);
}

/**
 * The recorded two-stage probe schedule: every iterate to `head`, then every `stride`-th.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.3
 */
export interface ProbeSchedule {
    head: number;
    stride: number;
}

/**
 * Spec §7.3's ruling: "every iterate to k=80, then every 20th". MEASURED, not chosen:
 * `e1-closer-0b-n*.json` `etaArgminIter` puts the `η` floor at iterations 36-60, so a uniform
 * stride of 20 would take 2-3 probes before the floor and could miss it. The gated value is a
 * MINIMUM over probes, so a sparser tail can only report a higher `η` — conservative, no false pass.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.3
 */
export const DEFAULT_PROBE_SCHEDULE: Readonly<ProbeSchedule> = { head: 80, stride: 20 };

function checkSchedule(s: ProbeSchedule): void {
    if (!Number.isInteger(s.head) || s.head < 0 || !Number.isInteger(s.stride) || s.stride < 1) {
        throw new RangeError(
            `gmres: probe schedule {head: ${s.head}, stride: ${s.stride}} needs integer head >= 0, stride >= 1`,
        );
    }
}

/**
 * The 1-based iteration indices a schedule probes within `maxit`: `1..head`, then
 * `head + stride, head + 2·stride, …`. At `maxit = 600` and the default schedule that is 80 + 26 =
 * 106 probes — spec §7.3's own cost arithmetic. Exposed as data because "with a recorded probe
 * stride" is in §11.2 row 7's requirement and an unrecorded schedule is unfalsifiable.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.3
 */
export function probeIterations(
    maxit: number,
    schedule: ProbeSchedule = DEFAULT_PROBE_SCHEDULE,
): number[] {
    checkSchedule(schedule);
    const out: number[] = [];
    for (let k = 1; k <= Math.min(schedule.head, maxit); k++) out.push(k);
    for (let k = schedule.head + schedule.stride; k <= maxit; k += schedule.stride) out.push(k);
    return out;
}

/**
 * `‖A‖₂` by power iteration on a symmetric matvec: `v ← A v / ‖A v‖`, the estimate after step `k`
 * being `‖A v_{k-1}‖` for unit `v_{k-1}`. For a symmetric operator that sequence is non-decreasing
 * and bounded above by `λ_max`, so the estimate is a LOWER bound that tightens — which is what
 * lets INVALID clause 4 (`‖A_σ‖₂_est < ‖A_σx̃‖/‖x̃‖`) be a pure comparison. Start vector:
 * deterministic LCG, uniform in `[-0.5, 0.5)`; spec §7.3's residual risk (a start vector nearly
 * orthogonal to the top block) is what clause 4 catches. 30 iterations suffice because the top
 * eigenvalues are nearly degenerate and the answer is correspondingly insensitive (spec §7.3).
 * @param matvec `A_σ` on `dim` entries; for H2 this is the HIERARCHICAL (θ) matvec, not χ=0.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.3
 */
export function estimateNorm2(
    matvec: Matvec,
    dim: number,
    iters: number = NORM_POWER_ITERS,
    seed: number = NORM_SEED,
): { norm: number; matvecs: number; history: number[] } {
    let s = seed >>> 0;
    const v = new Float64Array(dim);
    for (let i = 0; i < dim; i++) {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        v[i] = s / 4294967296 - 0.5;
    }
    let vn = norm(v);
    for (let i = 0; i < dim; i++) v[i] /= vn;
    const w = new Float64Array(dim);
    const history: number[] = [];
    for (let k = 0; k < iters; k++) {
        matvec(v, w);
        vn = norm(w);
        history.push(vn);
        for (let i = 0; i < dim; i++) v[i] = w[i] / vn;
    }
    return { norm: history[history.length - 1], matvecs: iters, history };
}

/**
 * The four INVALID clauses of spec §7.3, plus the one this file adds ({@link etaViolations}).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.3
 */
export type EtaClause =
    | 'eta-zero'
    | 'eta-below-uop'
    | 'exact-equals-bct'
    | 'norm-below-lower-bound'
    | 'non-finite';

/**
 * Which INVALID clauses of spec §7.3 one probe violates (empty ⇒ valid):
 *  1. `eta-zero` — `η == 0` exactly: the residual was formed with the operator that produced `x̃`.
 *  2. `eta-below-uop` — `0 < η < u_op/10`, `u_op` the roundoff of the OPERATOR's own precision, NOT
 *     `u_f32`: #59's unscoped `η < u_f32/10` rules every correct f64 run INVALID (spec §7.3).
 *  3. `exact-equals-bct` — `η_exact` BIT-equal to `η_bct`: both are computed and recorded.
 *  4. `norm-below-lower-bound` — `‖A_σ‖₂_est < ‖A_σx̃‖/‖x̃‖`: the estimate is below a free exact
 *     lower bound, so it never entered the top eigenspace.
 *  5. `non-finite` — NOT in the spec. NaN fails every `<` and `===` above, so a NaN `η` would read
 *     as valid on all four (memory `gates-name-their-impossible-values`).
 * Clauses 1 and 2 are disjoint (`η = 0` reports only clause 1).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.3
 */
export function etaViolations(
    p: { etaExact: number; etaBct: number; lowerBound: number; normA2: number },
    uOp: number,
): EtaClause[] {
    const out: EtaClause[] = [];
    if (!Number.isFinite(p.etaExact) || !Number.isFinite(p.etaBct)) out.push('non-finite');
    else {
        if (p.etaExact === 0) out.push('eta-zero');
        else if (p.etaExact < uOp / 10) out.push('eta-below-uop');
        if (p.etaExact === p.etaBct) out.push('exact-equals-bct');
    }
    if (p.normA2 < p.lowerBound) out.push('norm-below-lower-bound');
    return out;
}

/**
 * What {@link gmres} needs to probe `η`.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.3
 */
export interface EtaProbeOptions {
    /**
     * The χ=0 EXACT streaming `A_bct` (no shift): `A_σ x̃` is recomputed from `b` with this and
     * NEVER with the BCT used inside the iteration (spec §7.3). The shift is formed here from the
     * SAME `op.sigma` as the iteration's, so the two matvecs differ only in the BCT approximation.
     */
    exactApplyA: (u: ArrayLike<number>, out: Float64Array) => void;
    /**
     * Unit roundoff of the OPERATOR's own precision ({@link U_F32}, {@link U_F64}) — REQUIRED, with
     * no default, so it cannot silently be `u_f32` for an f64 run (INVALID clause 2).
     */
    uOp: number;
    /** `‖A_σ‖₂`. Default: {@link estimateNorm2} on the iteration's own hierarchical `A_σ`. */
    normA2?: number;
    /** Default {@link DEFAULT_PROBE_SCHEDULE}. Recorded in the result. */
    schedule?: ProbeSchedule;
    /** Stop at the first probe with `η_exact <= etaTarget` (H3's `τ_η`). Recorded; default none. */
    etaTarget?: number;
}

/**
 * One probed iterate. `iter` is 1-based: `x̃_iter` after `iter` Arnoldi steps.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.3
 */
export interface EtaProbeRecord {
    iter: number;
    /** `true` when the schedule did not cover this iterate but the solve stopped here. */
    terminal: boolean;
    xNorm: number;
    resNormExact: number;
    resNormBct: number;
    etaExact: number;
    etaBct: number;
    /** `‖A_σ x̃‖/‖x̃‖` by the exact matvec — the free lower bound of INVALID clause 4. */
    lowerBound: number;
    violations: EtaClause[];
}

/**
 * Everything the `η` probe measured and the schedule it measured it on.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.3, §11.2 task 7
 */
export interface EtaProbeReport {
    /** The schedule as configured. */
    schedule: ProbeSchedule;
    /** `probeIterations(maxit, schedule)` — what was PLANNED; `records[].iter` is what happened. */
    planned: number[];
    records: EtaProbeRecord[];
    normA2: number;
    /** 0 when `normA2` was supplied. */
    normA2Matvecs: number;
    uOp: number;
    etaTarget: number | null;
    /** Probe-only applies, SEPARATE from `GmresResult.matvecs`. Both equal `records.length`. */
    exactMatvecs: number;
    bctMatvecs: number;
    /** The gated value of H2: `min` over records of `η_exact`. */
    etaMin: number;
    etaMinIter: number;
    /** Union of the records' violations: non-empty ⇒ INVALID. */
    invalid: EtaClause[];
}

/**
 * Options of {@link gmres}.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.3, §7.3
 */
export interface GmresOptions {
    /**
     * Stop when the RECURRENCE residual is `<= tol`; `0` ⇒ never (not even at an exact 0, which
     * is reported as `'breakdown'`). REQUIRED, no default: spec §5.3
     * — "a parameter, recorded per solve, never a constant buried in the loop"; H2 and H3 pin
     * different values and both must be reachable. The recurrence is not a convergence criterion.
     */
    tol: number;
    /** Default {@link defaultMaxit}`(3n)`. Recorded in the result. */
    maxit?: number;
    /** Default `'left'`. `'right'` needs `knockOut: true` (module TSDoc). */
    side?: 'left' | 'right';
    /**
     * Lets `side: 'right'` through. MEASURED precedent for the guard: `precond.test.ts` was
     * 13 pass / 0 fail with a knock-out `L^{2-s}` as its default before its own guard existed.
     * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.3
     */
    knockOut?: boolean;
    probe?: EtaProbeOptions;
    /** Keep `V` and the un-rotated Hessenberg columns in the result (tests; O(maxit·dim) memory). */
    retainArnoldi?: boolean;
}

/**
 * Why the loop ended; `censored` is NOT derived from this (it is `iters === maxit`).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.3, §7.4
 */
export type GmresStop = 'tol' | 'eta-target' | 'breakdown' | 'maxit' | 'zero-rhs';

/**
 * Result of {@link gmres}: `x̃`, `b`, `iters`, `maxit`, `tol`, the per-iteration recurrence residual
 * and the `η` probe, exactly the list of spec §11.2 row 7.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.6, §11.2 task 7
 */
export interface GmresResult {
    /** `x̃` after `iters` iterations. */
    x: Float64Array;
    /** A copy of the right-hand side. */
    b: Float64Array;
    side: 'left' | 'right';
    iters: number;
    maxit: number;
    tol: number;
    stop: GmresStop;
    /**
     * `iters === maxit` — spec §11.2 row 7's literal predicate, compared against `maxit` and NEVER
     * `n`. Conservative by exactly one iteration: a criterion met AT `maxit` is also flagged, since
     * it cannot be told from a budget cap by the count alone (`stop` can). Under H2's run-to-floor
     * this is expected true and is not an INVALID trigger; under H3's count-to-`τ_η` it is (spec §7.4).
     */
    censored: boolean;
    /**
     * `|g_{k+1}| / β`, one per iteration: the Givens least-squares residual relative to `beta`
     * (`‖Ã⁻¹b‖` left, `‖b‖` right). NOT the true residual when left-preconditioned (module TSDoc).
     */
    recurrence: Float64Array;
    /** The norm `recurrence` is relative to. */
    beta: number;
    /** `A_σ` applications by the iteration itself (probe applies are in `probe`). */
    matvecs: number;
    /**
     * `Ã⁻¹` applications. Left: `iters + 1` (the rhs, then one per iteration), probes add none.
     * Right: `iters` plus ONE to form `x̃ = Ã⁻¹ V y` — per probe when probing (the stopping iterate
     * is always probed, so it is the terminal probe's), else once at the end.
     */
    precondApplies: number;
    probe?: EtaProbeReport;
    arnoldi?: { basis: Float64Array[]; hessenberg: Float64Array[] };
}

const norm = (x: ArrayLike<number>): number => {
    let s = 0;
    for (let i = 0; i < x.length; i++) s += x[i] * x[i];
    return Math.sqrt(s);
};

/**
 * Solves `A_σ x = b` by full GMRES with the injected preconditioner `precond` (module TSDoc).
 * @param op `A_bct` and `σ`; the shift is formed here ({@link applyAsigma}).
 * @param precond `Ã⁻¹` on `3n`; only `apply` and `n` are used.
 * @param b right-hand side, `3n`, coordinate-block layout.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.3, §5.6, §7.3
 */
export function gmres(
    op: HierarchicalOperator,
    precond: Pick<Preconditioner, 'apply' | 'n'>,
    b: ArrayLike<number>,
    opts: GmresOptions,
): GmresResult {
    const n = op.n;
    const dim = 3 * n;
    const side = opts.side ?? 'left';
    const { tol } = opts;
    if (b.length !== dim)
        throw new RangeError(`gmres: b has length ${b.length}, expected 3n = ${dim}`);
    if (precond.n !== n) {
        throw new RangeError(
            `gmres: the preconditioner was built for n=${precond.n}, the operator has n=${n}`,
        );
    }
    if (!(Number.isFinite(tol) && tol >= 0)) {
        throw new RangeError(
            `gmres: tol=${tol} must be finite and >= 0 (0 = never stop on the recurrence)`,
        );
    }
    const maxit = opts.maxit ?? defaultMaxit(dim);
    if (!Number.isInteger(maxit) || maxit < 1) {
        throw new RangeError(`gmres: maxit=${maxit} must be a positive integer`);
    }
    // Right preconditioning is a knock-out instrument: see the module TSDoc and spec §5.3.
    if (side === 'right' && !opts.knockOut) {
        throw new Error(
            "gmres: side 'right' is a knock-out instrument (spec §5.3: right-preconditioned f32 reads a healthy recurrence residual while the direction is destroyed); pass knockOut: true in a test",
        );
    }
    const probeOpts = opts.probe;
    if (probeOpts && !(probeOpts.uOp > 0 && Number.isFinite(probeOpts.uOp))) {
        throw new RangeError(`gmres: probe.uOp=${probeOpts.uOp} must be a positive unit roundoff`);
    }
    const schedule: ProbeSchedule = { ...(probeOpts?.schedule ?? DEFAULT_PROBE_SCHEDULE) };
    const planned = probeOpts ? probeIterations(maxit, schedule) : [];
    const plannedSet = new Set(planned);

    const bVec = Float64Array.from(b);
    const bNorm = norm(bVec);
    let matvecs = 0;
    let precondApplies = 0;
    const t = new Float64Array(dim);

    /** `out = M v`: left `Ã⁻¹ A_σ v`, right `A_σ Ã⁻¹ v`. */
    const applyM = (v: Float64Array, out: Float64Array): void => {
        if (side === 'left') {
            applyAsigma(op, v, t);
            matvecs++;
            precond.apply(t, out);
            precondApplies++;
        } else {
            precond.apply(v, t);
            precondApplies++;
            applyAsigma(op, t, out);
            matvecs++;
        }
    };

    const r0 = new Float64Array(dim);
    if (side === 'left') {
        precond.apply(bVec, r0);
        precondApplies++;
    } else {
        r0.set(bVec);
    }
    const beta = norm(r0);
    const finish = (
        x: Float64Array,
        iters: number,
        stop: GmresStop,
        recurrence: Float64Array,
        extra: Pick<GmresResult, 'probe' | 'arnoldi'>,
    ): GmresResult => ({
        x,
        b: bVec,
        side,
        iters,
        maxit,
        tol,
        stop,
        censored: iters === maxit,
        recurrence,
        beta,
        matvecs,
        precondApplies,
        ...extra,
    });
    if (bNorm === 0) return finish(new Float64Array(dim), 0, 'zero-rhs', new Float64Array(0), {});
    if (!(beta > 0 && Number.isFinite(beta))) {
        throw new Error(`gmres: ‖preconditioned rhs‖=${beta}; the preconditioner annihilated b`);
    }

    // --- the η probe's fixed ingredients -----------------------------------------------------
    let normA2 = Number.NaN;
    let normA2Matvecs = 0;
    const ax = new Float64Array(dim);
    const records: EtaProbeRecord[] = [];
    let bctMatvecs = 0;
    let exactMatvecs = 0;
    if (probeOpts) {
        if (probeOpts.normA2 !== undefined) {
            normA2 = probeOpts.normA2;
        } else {
            // The HIERARCHICAL matvec, not χ=0: spec §7.3's affordability ruling (30 χ=0 matvecs
            // is 2.86 h at N=30 000), justified by `lamMaxBct/lamMaxA` deviating <= 0.53 %.
            const est = estimateNorm2((u, out) => applyAsigma(op, u, out), dim);
            normA2 = est.norm;
            normA2Matvecs = est.matvecs;
        }
    }

    const cs = new Float64Array(maxit);
    const sn = new Float64Array(maxit);
    const g = new Float64Array(maxit + 1);
    g[0] = beta;
    const V: Float64Array[] = [];
    const first = new Float64Array(dim);
    for (let i = 0; i < dim; i++) first[i] = r0[i] / beta;
    V.push(first);
    const R: Float64Array[] = []; // rotated (upper-triangular) columns
    const Hraw: Float64Array[] = []; // un-rotated columns, kept only for `retainArnoldi`
    const recurrence = new Float64Array(maxit);
    const y = new Float64Array(maxit);

    /** `x̃` after `k` iterations: back-substitute `R y = g`, `x = V y` (right: `Ã⁻¹ V y`). */
    const solution = (k: number): Float64Array => {
        for (let i = k - 1; i >= 0; i--) {
            let s = g[i];
            for (let j = i + 1; j < k; j++) s -= R[j][i] * y[j];
            y[i] = s / R[i][i];
        }
        const u = new Float64Array(dim);
        for (let j = 0; j < k; j++) {
            const vj = V[j];
            const yj = y[j];
            for (let i = 0; i < dim; i++) u[i] += yj * vj[i];
        }
        if (side === 'left') return u;
        const x = new Float64Array(dim);
        precond.apply(u, x);
        precondApplies++;
        return x;
    };

    /** Forms `η` at `x` with the EXACT matvec and, separately, with the iteration's own BCT one. */
    const probeAt = (iter: number, terminal: boolean, x: Float64Array): EtaProbeRecord => {
        const xNorm = norm(x);
        // exact: χ=0 `A_bct` + the SAME shift, recomputed from `b` (spec §7.3).
        if (probeOpts) shiftedApply(probeOpts.exactApplyA, n, op.sigma, x, ax);
        exactMatvecs++;
        const axNorm = norm(ax);
        let rs = 0;
        for (let i = 0; i < dim; i++) rs += (bVec[i] - ax[i]) ** 2;
        const resNormExact = Math.sqrt(rs);
        // bct: the matvec the iteration used.
        applyAsigma(op, x, ax);
        bctMatvecs++;
        rs = 0;
        for (let i = 0; i < dim; i++) rs += (bVec[i] - ax[i]) ** 2;
        const resNormBct = Math.sqrt(rs);
        const rec: EtaProbeRecord = {
            iter,
            terminal,
            xNorm,
            resNormExact,
            resNormBct,
            etaExact: backwardError(resNormExact, normA2, xNorm, bNorm),
            etaBct: backwardError(resNormBct, normA2, xNorm, bNorm),
            lowerBound: xNorm > 0 ? axNorm / xNorm : 0,
            violations: [],
        };
        rec.violations = etaViolations({ ...rec, normA2 }, probeOpts?.uOp ?? Number.NaN);
        records.push(rec);
        return rec;
    };

    let iters = 0;
    let stop: GmresStop = 'maxit';
    let x: Float64Array | null = null;
    for (let k = 0; k < maxit; k++) {
        const w = new Float64Array(dim);
        applyM(V[k], w);
        const wNorm0 = norm(w);
        const h = new Float64Array(k + 2);
        // Modified Gram-Schmidt, `i = 0..k` INCLUSIVE: leaving out `v_k` itself destroys both the
        // orthogonality and the Hessenberg shape (gmres_lab.py:263-265).
        for (let i = 0; i <= k; i++) {
            const vi = V[i];
            let d = 0;
            for (let q = 0; q < dim; q++) d += vi[q] * w[q];
            h[i] = d;
            for (let q = 0; q < dim; q++) w[q] -= d * vi[q];
        }
        const hsub = norm(w);
        h[k + 1] = hsub;
        const breakdown = hsub <= BREAKDOWN_REL * wNorm0;
        if (opts.retainArnoldi) Hraw.push(Float64Array.from(h));
        if (!breakdown) {
            const next = new Float64Array(dim);
            for (let q = 0; q < dim; q++) next[q] = w[q] / hsub;
            V.push(next);
        }
        for (let i = 0; i < k; i++) {
            const tt = cs[i] * h[i] + sn[i] * h[i + 1];
            h[i + 1] = -sn[i] * h[i] + cs[i] * h[i + 1];
            h[i] = tt;
        }
        const d = Math.hypot(h[k], h[k + 1]);
        cs[k] = d ? h[k] / d : 1;
        sn[k] = d ? h[k + 1] / d : 0;
        h[k] = d;
        h[k + 1] = 0;
        g[k + 1] = -sn[k] * g[k];
        g[k] = cs[k] * g[k];
        R.push(h);
        recurrence[k] = Math.abs(g[k + 1]) / beta;
        iters = k + 1;

        // Stop rules, in this order: recurrence tol, then η target (below), then breakdown.
        // `tol > 0 &&`: tol = 0 means "no recurrence stop" (GmresOptions.tol), even at the instant
        // the recurrence reads exactly 0 — that case is a breakdown and is reported as one.
        const stopTol = tol > 0 && recurrence[k] <= tol;
        const stopNow = stopTol || breakdown || iters === maxit;
        if (probeOpts && (plannedSet.has(iters) || stopNow)) {
            x = solution(iters);
            const rec = probeAt(iters, !plannedSet.has(iters), x);
            if (probeOpts.etaTarget !== undefined && rec.etaExact <= probeOpts.etaTarget) {
                stop = 'eta-target';
                break;
            }
        } else {
            x = null;
        }
        if (stopTol) {
            stop = 'tol';
            break;
        }
        if (breakdown) {
            stop = 'breakdown';
            break;
        }
    }
    // `x` is the probed one when the last iteration was probed; otherwise form it now.
    const xTilde = x ?? solution(iters);

    let probe: EtaProbeReport | undefined;
    if (probeOpts) {
        let etaMin = Number.POSITIVE_INFINITY;
        let etaMinIter = 0;
        const invalid = new Set<EtaClause>();
        for (const rec of records) {
            if (rec.etaExact < etaMin || Number.isNaN(rec.etaExact)) {
                etaMin = rec.etaExact;
                etaMinIter = rec.iter;
            }
            for (const c of rec.violations) invalid.add(c);
        }
        probe = {
            schedule,
            planned,
            records,
            normA2,
            normA2Matvecs,
            uOp: probeOpts.uOp,
            etaTarget: probeOpts.etaTarget ?? null,
            exactMatvecs,
            bctMatvecs,
            etaMin,
            etaMinIter,
            invalid: [...invalid],
        };
    }
    return finish(xTilde, iters, stop, recurrence.slice(0, iters), {
        probe,
        arnoldi: opts.retainArnoldi ? { basis: V, hessenberg: Hraw } : undefined,
    });
}
