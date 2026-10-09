/**
 * `Ã⁻¹ = (−Δ)⁻¹ L^{2−s} (−Δ)⁻¹` — the Repulsive Surfaces preconditioner (`[RS §5.2]`,
 * arXiv:2107.01664), with the constant-mode correction, for left-preconditioned GMRES (spec §5.3).
 *
 * THREE OPERATOR APPLICATIONS PER CALL, ALL O(N): two cyclic-tridiagonal back-substitutions
 * (`laplacian.ts` `solveLaplacian`, Sherman-Morrison / NR §2.7) around ONE `L^{2−s}` product
 * (`lowOrder.ts`, 3 columns on the same tree and the same block lists as `A`). No matrix is ever
 * formed; a dense `Ã⁻¹` exists only in tests and is labelled there.
 *
 * THE STRUCTURE, `.superpowers/sdd/2026-09-11-phase2h-queue/scratch/gmres_lab.py:174-194`
 * (`rs_preconditioner`) verbatim:
 *
 *     Π      = I − 1 1ᵀ / n                      the projector off the constants
 *     core   = Π D⁺ L^{2−s} D⁺ Π                 with D⁺ = pinv(Π (−Δ) Π) in the lab
 *     Ã⁻¹    = core + 1 1ᵀ / (n σ)
 *
 * and since `D⁺ = Π D⁺ Π`, `core` expands to `Π S Π L^{2−s} Π S Π` — which is what
 * {@link applyPrecondColumns} evaluates left to right, with `S` the production solve.
 *
 * WHY `Π` APPEARS ON BOTH SIDES, and why the `+11ᵀ/(nσ)` term is exactly that. Both `(−Δ)` (pure
 * stiffness) and `L^{2−s}` are singular on the constants — exactly as `A` is before the rank-1
 * shift — so the RS-faithful operator lives on `1⊥` and `Π` is what keeps it there. `A_σ`'s
 * constant mode has eigenvalue **exactly σ**, because `A·1 = 0` bit-exactly (spec §2.2,
 * `apply.ts`) and `(σ/n)·1 1ᵀ·1 = σ·1`. So `+11ᵀ/(nσ)` makes the preconditioner EXACT on the
 * constant mode — `Ã⁻¹A_σ 1 = 1` — while staying RS-faithful on its complement. Delete it and `Ã`
 * is singular on the constants, `Ã⁻¹A_σ` loses rank 1 and `κ_eig` is infinite: that is spec §7.8's
 * knock-out (2), reachable here through {@link PreconditionerOptions.dropConstantMode}.
 *
 * `σ` MUST BE THE STRUCTURAL `tr(A_bct)/n` of `operator.ts` (`HierarchicalOperator.sigma`). Spec
 * §2.3 forbids an exact trace in production (densifying is `n` applies, ~64 min at n=30 000).
 *
 * ONE DEVIATION FROM THE MEASURED ORACLE, RECORDED BECAUSE IT CROSSES A PROVENANCE BOUNDARY. The
 * lab's `D⁺` is `pinv(Π Lap Π)` of the PURE-STIFFNESS polyline Laplacian (`gmres_lab.py:95-110`,
 * `w = 1/ell`, singular with nullspace the constants). Production uses `solveLaplacian` on
 * `stiffness·L + mass·M` with **`massWeight = 1`** — spec §11.2's mass-weight ruling, because
 * `massWeight <= 0` is singular and has no pseudo-inverse route (`laplacian.ts` throws) and 1e-3
 * breached 7 of 9 measured cases. These are DIFFERENT operators, so every `κ_eig` in
 * `e0-closer-0b.json` / `e4-closer-0b.json` characterises the LAB's `Ã⁻¹`, not this one.
 * `massWeight` therefore stays a runtime parameter and `precond.test.ts` measures how far the
 * production operator moves as it varies. It now ALSO measures `κ_eig` of THIS operator directly
 * (`denseSpectrum.ts`, validated against the corpus to 8 digits on the lab construction), so the
 * two no longer have to be conflated. Two consequences a caller must know: the shipped `κ` is
 * better than the lab's, and it is NOT scale-invariant — `massWeight = 1` carries units of
 * `length⁻²` (stiffness scales as `length⁻¹`, mass as `length⁺¹`), and `κ` moves 53-62x between
 * bbox extent 5.4716 and extent 1.0. A `κ` quoted without its extent is not a measurement.
 *
 * NOT RE-ENTRANT, for the same reason `HierarchicalA` is not (`apply.ts`): the workspace, the
 * scratch and the wrapped `LowOrderA` are shared, and GMRES calls this inside its loop.
 *
 * @see https://arxiv.org/abs/2107.01664 §5.2
 * @see .superpowers/sdd/2026-09-11-phase2h-queue/scratch/gmres_lab.py:174-194
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §2.3, §5.2, §5.3, §7.8, §11.2 task 6
 * @issue utof/repulsive-test2#59
 */
import type { Edge, Vec3 } from '../../testConfigs';
import {
    assembleCurveLaplacian,
    type LaplacianWorkspace,
    makeLaplacianWorkspace,
    solveLaplacian,
    type TriSystem,
} from './laplacian';
import { applyLowOrderColumns, type LowOrderA } from './lowOrder';

/**
 * Options of {@link makePreconditioner}.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.8, §11.2
 */
export interface PreconditionerOptions {
    /** `(−Δ)`'s stiffness weight. Default 1. */
    stiffnessWeight?: number;
    /**
     * `(−Δ)`'s mass weight. Default **1** — spec §11.2's mass-weight ruling (`<= 0` is singular,
     * 1e-3 breached 7 of 9 measured cases). See the module TSDoc's provenance note.
     */
    massWeight?: number;
    /**
     * KNOCK-OUT INSTRUMENT, tests and gate calibration only. Drops the constant-mode correction
     * `+11ᵀ/(n·σ)` ⇒ `Ã` singular on the constants ⇒ `Ã⁻¹A_σ` loses rank 1 ⇒ H5 property 1 RED.
     * Spec §7.8 knock-out (2), which it records as "NOT MEASURED, cheap"; `precond.test.ts`
     * measures it. Do not delete as dead code: a knock-out that cannot fire is the defect this
     * phase has hit twice (memory `gates-name-their-impossible-values`).
     * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.8
     */
    dropConstantMode?: boolean;
    /**
     * Lets a knock-out `L^{2−s}` through — currently the `'rs-identity'` variant, whose nullity is
     * 2 on an even closed polyline (`lowOrder.ts`'s `LowOrderVariant`). Without this flag
     * {@link makePreconditioner} REFUSES it, mirroring `prepareA` (`apply.ts:482-490`, commit
     * `2a8b88b`): a configuration the code itself calls a knock-out must not be
     * production-reachable, because `Ã⁻¹` built on it kills the mode `A_σ` weights MOST and every
     * preconditioner-level test still reads GREEN (fix round 1, finding I3).
     * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §7.8
     */
    knockOut?: boolean;
}

/**
 * A prepared `Ã⁻¹`. NOT RE-ENTRANT (module TSDoc).
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §5.2
 */
export interface Preconditioner {
    /** Vertex count `|V|`; the 3n operator is three copies of the scalar one. */
    n: number;
    /** `tr(A_bct)/n`, as handed in — the shift of `A_σ = A + (σ/n) 11ᵀ`. */
    sigma: number;
    /** The assembled `(−Δ)` bands; exposed so tests can check the solve with `triResidual`. */
    sys: TriSystem;
    lowOrder: LowOrderA;
    /** `true` when the constant-mode correction is dropped (knock-out). */
    dropConstantMode: boolean;
    /** `out = Ã⁻¹ R` on `3n` in the coordinate-block layout `R[c*n + i]`. */
    apply(R: ArrayLike<number>, out: Float64Array): void;
    /** `out = Ã⁻¹ r` for a scalar vertex function (length `n`). */
    applyScalar(r: ArrayLike<number>, out: Float64Array): void;
    /**
     * Cumulative `solveLaplacian` calls — EXACTLY `2 * nb` per apply. A structural counter, not a
     * diagnostic: "three operator applications" is the O(N) claim of §5.2, and a fourth solve
     * (or a dense route) would read exactly like a passing test without it.
     */
    solves: number;
    /** Cumulative `L^{2−s}` applications — EXACTLY 1 per apply, for the same reason. */
    lowOrderApplies: number;
}

/**
 * `(Π x)_i = x_i − mean(x)` — the projector off the constants (module TSDoc), with a NEUMAIER
 * compensated sum and TWO passes.
 *
 * Neither is a micro-optimisation, and this is the line a future agent would "clean up". A naive
 * single-pass `sum/n` gets the mean wrong by O(n·u)·|x|, so `Π` leaves a residual constant
 * component which the two `(−Δ)⁻¹` solves then amplify — landing on exactly the mode the
 * `+11ᵀ/(n·σ)` correction exists to make exact.
 *
 * WHY IT IS KEPT — the structural zeros, NOT the conditioning. A constant-mode error is
 * Krylov-IRRELEVANT: the constant mode is a single eigenvalue pinned at 1 and GMRES disposes of it
 * in the first iteration, so even 1e-2 of error there would not cost an iteration. What the
 * compensated sum buys is that `§7.8`'s knock-out (2) reads EXACTLY 0 instead of ~1e-11, which is
 * what makes `precond.test.ts`'s `expect(...).toBe(0)` a structural verdict rather than a
 * tolerance — and per the round-1 review that `toBe(0)` is the only thing in the suite that gates
 * this function at all.
 *
 * MEASURED, naive vs this, `max|Ã⁻¹A_σ·1 − 1|`: 1.75e-13 → 1.11e-15 at n=64 and
 * 2.07e-11 → 2.22e-15 at n=240 — a 9 400x improvement at n=240, and FLAT in n instead of growing
 * with the naive sum's fitted exponent of **4.29** (6 points, n=64..1920; 4.38 over the four
 * largest), which extrapolates to **~3e-2 at the n=30 000 target**. The `n^3.6` in the first draft
 * of this comment was a two-point ratio and understated it. Do not "simplify" this back to one
 * uncompensated pass.
 * @see https://doi.org/10.1002/zamm.19740540106 (Neumaier 1974, the compensated sum)
 */
function project(x: Float64Array, n: number): void {
    for (let pass = 0; pass < 2; pass++) {
        let s = 0;
        let c = 0;
        for (let i = 0; i < n; i++) {
            const t = s + x[i];
            c += Math.abs(s) >= Math.abs(x[i]) ? s - t + x[i] : x[i] - t + s;
            s = t;
        }
        const mean = (s + c) / n;
        if (mean === 0) return; // already exactly mean-free; the second pass has nothing to do.
        for (let i = 0; i < n; i++) x[i] -= mean;
    }
}

interface PrecondState {
    n: number;
    sigma: number;
    sys: TriSystem;
    lowOrder: LowOrderA;
    dropConstantMode: boolean;
    ws: LaplacianWorkspace;
    /** `nb`-block scratch: `mid` = `Π S Π R`, `lz` = `L^{2−s} mid`; `a`/`b` = one solve's pair. */
    mid: Float64Array;
    lz: Float64Array;
    a: Float64Array;
    b: Float64Array;
    counters: { solves: number; lowOrderApplies: number };
}

/**
 * `out = Ã⁻¹ R` for `nb` coordinate blocks. `A_σ` is block-diagonal over the three components
 * (`src/core/sobolev/layout.ts` `expandBlockDiag`), so the two solves run per block while the
 * `L^{2−s}` product is ONE `nb`-column apply — spec §5.2's *"3 columns on the same tree"*.
 *
 * Operand order is `Π S Π · L^{2−s} · Π S Π` and is NOT decoration: `Π` on the inner sides keeps
 * the solve's output in `1⊥` where the RS operator is defined, and `Π` on the outer sides makes
 * the core orthogonal to the `+11ᵀ/(nσ)` term so the two cannot interfere on the constant mode.
 * The two `Π`s around `L^{2−s}` are redundant to roundoff, because `L^{2−s}·1 = 0` holds
 * bit-exactly FOR THE ONES VECTOR (`lowOrder.ts`; `L·(7.5·1)` is 3e-14..1.2e-13, not 0, so this is
 * not a claim about general constant vectors) and hence `ΠL = LΠ = L` on the range that reaches
 * them. They are kept because the lab's `core` is `Π D⁺ L D⁺ Π` and `D⁺ = Π D⁺ Π`, so writing
 * the solve's `Π`s out is how `Π S Π L Π S Π` matches `gmres_lab.py:190-192`'s
 * `Pi @ Dplus @ Lfrac @ Dplus @ Pi` — which has only the OUTER pair. Deleting the inner pair is
 * a measured no-op; deleting an OUTER one is not (`precond.test.ts`'s leakage test).
 */
function applyPrecondColumns(
    P: PrecondState,
    R: ArrayLike<number>,
    out: Float64Array,
    nb: number,
): void {
    const { n, sys, ws, mid, lz, a, b, lowOrder, sigma } = P;

    for (let blk = 0; blk < nb; blk++) {
        const o = blk * n;
        for (let i = 0; i < n; i++) a[i] = R[o + i];
        project(a, n);
        solveLaplacian(sys, a, b, ws);
        P.counters.solves++;
        project(b, n);
        for (let i = 0; i < n; i++) mid[o + i] = b[i];
    }

    applyLowOrderColumns(lowOrder, mid, lz, nb);
    P.counters.lowOrderApplies++;

    for (let blk = 0; blk < nb; blk++) {
        const o = blk * n;
        for (let i = 0; i < n; i++) a[i] = lz[o + i];
        project(a, n);
        solveLaplacian(sys, a, b, ws);
        P.counters.solves++;
        project(b, n);
        // The constant-mode correction `+11ᵀ/(n σ)` (module TSDoc: `A_σ`'s constant mode has
        // eigenvalue exactly σ, so this makes `Ã⁻¹` exact there). Dropping it is §7.8's
        // knock-out (2), NOT an optimisation — `Ã` is then singular on the constants.
        let c = 0;
        if (!P.dropConstantMode) {
            let s = 0;
            for (let i = 0; i < n; i++) s += R[o + i];
            c = s / (n * sigma);
        }
        for (let i = 0; i < n; i++) out[o + i] = b[i] + c;
    }
}

/**
 * Builds a {@link Preconditioner} over an existing `L^{2−s}` (which already carries the tree and
 * the block lists `A` was prepared from) and the structural `σ = tr(A_bct)/n` of `operator.ts`.
 *
 * @param vertices/edges the curve, for `assembleCurveLaplacian`; its topology limit (natural open
 *   path or closed ring) is the preconditioner's, and it throws `CurveTopologyError` otherwise.
 * @param sigma MUST be `HierarchicalOperator.sigma` — spec §2.3 forbids an exact trace in
 *   production. Rejected unless finite and `> 0`: a zero or NaN `σ` would turn the constant-mode
 *   correction into `Infinity`/`NaN` that GMRES would carry silently.
 * @see docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md §2.3, §5.2
 */
export function makePreconditioner(
    vertices: Vec3[],
    edges: Edge[],
    lowOrder: LowOrderA,
    sigma: number,
    opts: PreconditionerOptions = {},
): Preconditioner {
    const n = vertices.length;
    if (!(sigma > 0) || !Number.isFinite(sigma)) {
        throw new Error(
            `makePreconditioner: sigma=${sigma} is not a usable shift; pass HierarchicalOperator.sigma (= tr(A_bct)/n, spec §2.3)`,
        );
    }
    if (lowOrder.n !== n) {
        throw new Error(
            `makePreconditioner: L^{2-s} was prepared for n=${lowOrder.n}, the curve has n=${n}`,
        );
    }
    // Knock-out instruments are refused unless explicitly let through, exactly as `prepareA` does
    // (`apply.ts:482-490`). `'rs-identity'` has nullity 2 for even n, so `Ã` is singular on the
    // checkerboard and `Ã⁻¹` stops being a preconditioner — and `precond.test.ts` used to pass in
    // full with it as the default (fix round 1, finding I3).
    if (!opts.knockOut && lowOrder.variant !== 'vertex') {
        throw new Error(
            `makePreconditioner: knock-out L^{2-s} in a production preconditioner (variant=${lowOrder.variant}); pass knockOut: true in a test`,
        );
    }
    const sys = assembleCurveLaplacian(
        vertices,
        edges,
        opts.stiffnessWeight ?? 1,
        opts.massWeight ?? 1,
    );
    const state: PrecondState = {
        n,
        sigma,
        sys,
        lowOrder,
        dropConstantMode: opts.dropConstantMode ?? false,
        ws: makeLaplacianWorkspace(n),
        mid: new Float64Array(3 * n),
        lz: new Float64Array(3 * n),
        a: new Float64Array(n),
        b: new Float64Array(n),
        counters: { solves: 0, lowOrderApplies: 0 },
    };
    return {
        n,
        sigma,
        sys,
        lowOrder,
        dropConstantMode: state.dropConstantMode,
        apply: (R, out) => applyPrecondColumns(state, R, out, 3),
        applyScalar: (r, out) => applyPrecondColumns(state, r, out, 1),
        get solves() {
            return state.counters.solves;
        },
        get lowOrderApplies() {
            return state.counters.lowOrderApplies;
        },
    };
}
